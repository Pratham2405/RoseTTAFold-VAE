import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m
from RoseTTAFoldModel import RoseTTAFoldModule_e2e
import util
from collections import namedtuple
from trFold import TRFold
from kinematics import PARAMS

script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
NBIN = [37, 37, 37, 19]

MODEL_PARAM ={
    "n_module"      : 8,
    "n_module_str"  : 4,
    "n_module_ref"  : 4,
    "n_layer"       : 1,
    "d_msa"         : 384,
    "d_pair"        : 288,
    "d_templ"       : 64,
    "n_head_msa"    : 12,
    "n_head_pair"   : 8,
    "n_head_templ"  : 4,
    "d_hidden"      : 64,
    "r_ff"          : 4,
    "n_resblock"    : 1,
    "p_drop"        : 0.0,
    "use_templ"     : True,
    "performer_N_opts": {"nb_features": 64},
    "performer_L_opts": {"nb_features": 64}
}

SE3_param = {
    "num_layers"        : 2,
    "num_channels"      : 16,
    "num_degrees"       : 2,
    "l0_in_features"    : 32,
    "l0_out_features"   : 8,
    "l1_in_features"    : 3,
    "l1_out_features"   : 3,
    "num_edge_features" : 32,
    "div"               : 2,
    "n_heads"           : 4
}

REF_param = {
    "num_layers"        : 3,
    "num_channels"      : 32,
    "num_degrees"       : 3,
    "l0_in_features"    : 32,
    "l0_out_features"   : 8,
    "l1_in_features"    : 3,
    "l1_out_features"   : 3,
    "num_edge_features" : 32,
    "div"               : 4,
    "n_heads"           : 4
}

MODEL_PARAM['SE3_param'] = SE3_param
MODEL_PARAM['REF_param'] = REF_param

# params for the folding protocol
fold_params = {
    "SG7"    : np.array([[[-2,3,6,7,6,3,-2]]])/21,
    "SG9"    : np.array([[[-21,14,39,54,59,54,39,14,-21]]])/231,
    "DCUT"   : 19.5,
    "ALPHA"  : 1.57,
    "NCAC"   : np.array([[-0.676, -1.294, 0.],
                         [ 0.    , 0.    , 0.],
                         [ 1.5   ,-0.174 , 0.]], dtype=np.float32),
    "CLASH"  : 2.0,
    "PCUT"   : 0.5,
    "DSTEP"  : 0.5,
    "ASTEP"  : np.deg2rad(10.0),
    "XYZRAD" : 7.5,
    "WANG"   : 0.1,
    "WCST"   : 0.1
}

fold_params["SG"] = fold_params["SG9"]


def c6d_to_t2d(c6d_array, t0d_confidence=0.7, params=None):
    '''
    Convert c6d representations to t2d format for RoseTTAFold.
    
    Args:
        c6d_array: numpy array of shape (B, L, L, 4) 
                   containing [dist, omega, theta, phi] for each residue pair
        t0d_confidence: confidence score to assign to t0d and t1d (default: 0.7)
        params: optional dict with 'DMAX' key (defaults to 20.0)
    
    Returns:
        t2d: torch tensor of shape (1, B, L, L, 10) - 2D template features
        t1d: torch tensor of shape (1, B, L, 3) - 1D template features  
        t0d: torch tensor of shape (1, B, 3) - 0D template features
    
    Notes:
        - B represents number of templates
        - Output has 10 dims: 1 (dist) + 6 (sin/cos of 3 angles) + 3 (t0d features)
        - The model internally projects these 10 dims to d_templ=64
    '''
    if params is None:
        params = {'DMAX': 20.0}
    
    # Convert to tensor if numpy
    if isinstance(c6d_array, np.ndarray):
        c6d = torch.from_numpy(c6d_array).float()
    else:
        c6d = c6d_array.float()
    
    B, L, _, _ = c6d.shape
    
    # Add batch dimension (script expects batch=1)
    # Shape becomes: (1, B, L, L, 4) where B is now n_templates
    c6d = c6d.unsqueeze(0)  # (1, B, L, L, 4)
    
    # Create mask for valid distances
    dist_raw = c6d[..., :1]  # (1, B, L, L, 1)
    mask = torch.isfinite(dist_raw).float()
    mask = mask * (dist_raw <= params['DMAX']).float()
    
    # Optionally zero out diagonal (self-interactions)
    eye = torch.eye(L, device=c6d.device, dtype=c6d.dtype).view(1, 1, L, L, 1)
    mask = mask * (1.0 - eye)
    
    # Normalize distance to [0, 1]
    dist = (c6d[..., :1] * mask) / params['DMAX']  # (1, B, L, L, 1)
    dist = torch.clamp(dist, 0.0, 1.0)
    
    # Convert angles to sin/cos representation
    # c6d[..., 1:] contains [omega, theta, phi]
    angles = c6d[..., 1:]  # (1, B, L, L, 3)
    orien = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) * mask  # (1, B, L, L, 6)
    
    # Create t0d (template-level features) and expand to pairwise
    t0d = torch.ones((1, B, 3)).float() * t0d_confidence  # (1, B, 3)
    t0d_pair = t0d.unsqueeze(2).unsqueeze(3).expand(-1, -1, L, L, -1)  # (1, B, L, L, 3)
    
    # Concatenate: 1 (dist) + 6 (orientation) + 3 (t0d) = 10 dims
    t2d = torch.cat([dist, orien, t0d_pair], dim=-1)  # (1, B, L, L, 10)
    t2d[torch.isnan(t2d)] = 0.0
    
    # Create t1d (per-residue template features)
    t1d = torch.ones((1, B, L, 3)).float() * t0d_confidence  # (1, B, L, 3)
    
    return t2d, t1d, t0d



class Predictor():
    def __init__(self, model_dir=None, use_cpu=False):
        if model_dir == None:
            self.model_dir = "%s/models"%(os.path.dirname(os.path.realpath(__file__)))
        else:
            self.model_dir = model_dir

        # define model name
        self.model_name = "RoseTTAFold"
        if torch.cuda.is_available() and (not use_cpu):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = RoseTTAFoldModule_e2e(**MODEL_PARAM).to(self.device)

    def load_model(self, model_name, suffix='e2e'):
        chk_fn = "%s/%s_%s.pt"%(self.model_dir, model_name, suffix)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return True

    def predict(self, a3m_fn, out_prefix, c6d_fn=None, t0d_confidence=0.7, 
                window=150, shift=75):
        '''
        Predict protein structure from MSA and optional c6d template features.

        Args:
            a3m_fn: Path to MSA file in a3m format (use minimal/single sequence for template-only)
            out_prefix: Output file prefix
            c6d_fn: Path to numpy file containing c6d array (n_templates, L, L, 6)
            t0d_confidence: Confidence score for template features (default: 0.7)
            window: Window size for cropped predictions (default: 150)
            shift: Shift size for sliding window (default: 75)
        '''

        # Parse MSA (can be minimal with just query sequence)
        msa = parse_a3m(a3m_fn)
        N, L = msa.shape

        print(f"Loaded MSA with {N} sequences and length {L}")

        # Handle template features
        if c6d_fn is not None:
            print(f"Loading c6d features from {c6d_fn}")
            c6d_array = np.load(c6d_fn)  # Expected shape: (n_templates, L, L, 6)

            # Verify shape compatibility
            if c6d_array.shape[1] != L or c6d_array.shape[2] != L:
                raise ValueError(f"c6d sequence length ({c6d_array.shape[1]}) does not match MSA length ({L})")

            # Convert c6d to t2d format
            t2d, t1d, t0d = c6d_to_t2d(c6d_array, t0d_confidence=t0d_confidence)
            print(f"Converted c6d to t2d with confidence={t0d_confidence}")
            print(f"Template shapes - t2d: {t2d.shape}, t1d: {t1d.shape}, t0d: {t0d.shape}")
        else:
            # No templates provided - use zeros
            print("No templates provided, using zero templates")
            t2d = torch.zeros((1, 1, L, L, 10)).float()
            t1d = torch.zeros((1, 1, L, 3)).float()
            t0d = torch.zeros((1, 1, 3)).float()

        # Prepare MSA tensors
        msa = torch.tensor(msa).long().view(1, -1, L)
        idx_pdb = torch.arange(L).long().view(1, L)
        seq = msa[:,0]

        # Load model
        could_load = self.load_model(self.model_name, suffix="e2e")
        if not could_load:
            print("ERROR: failed to load model")
            sys.exit()

        self.model.eval()
        with torch.no_grad():
            # Do cropped prediction if protein is too big
            if L > window*2:
                print(f"Protein length {L} > {window*2}, using windowed prediction")
                prob_s = [np.zeros((L,L,NBIN[i]), dtype=np.float32) for i in range(4)]
                count_1d = np.zeros((L,), dtype=np.float32)
                count_2d = np.zeros((L,L), dtype=np.float32)
                node_s = np.zeros((L,MODEL_PARAM['d_msa']), dtype=np.float32)

                grids = np.arange(0, L-window+shift, shift)
                ngrids = grids.shape[0]
                print("ngrid: ", ngrids)
                print("grids: ", grids)
                print("windows: ", window)

                for i in range(ngrids):
                    for j in range(i, ngrids):
                        start_1 = grids[i]
                        end_1 = min(grids[i]+window, L)
                        start_2 = grids[j]
                        end_2 = min(grids[j]+window, L)

                        sel = np.zeros((L)).astype(np.bool_)
                        sel[start_1:end_1] = True
                        sel[start_2:end_2] = True

                        input_msa = msa[:,:,sel]
                        mask = torch.sum(input_msa==20, dim=-1) < 0.5*sel.sum()
                        input_msa = input_msa[mask].unsqueeze(0)
                        input_msa = input_msa[:,:1000].to(self.device)
                        input_idx = idx_pdb[:,sel].to(self.device)
                        input_seq = input_msa[:,0].to(self.device)

                        # Select template regions
                        input_t1d = t1d[:,:,sel].to(self.device)
                        input_t2d = t2d[:,:,sel][:,:,:,sel].to(self.device)

                        print("running crop: %d-%d/%d-%d"%(start_1, end_1, start_2, end_2), input_msa.shape)
                        with torch.cuda.amp.autocast():
                            logit_s, node, init_crds, pred_lddt = self.model(
                                input_msa, input_seq, input_idx, 
                                t1d=input_t1d, t2d=input_t2d, return_raw=True)

                        sub_idx = input_idx[0].cpu()
                        sub_idx_2d = np.ix_(sub_idx, sub_idx)
                        count_2d[sub_idx_2d] += 1.0
                        count_1d[sub_idx] += 1.0
                        node_s[sub_idx] += node[0].cpu().numpy()

                        for i_logit, logit in enumerate(logit_s):
                            prob = self.active_fn(logit.float())
                            prob = prob.squeeze(0).permute(1,2,0).cpu().numpy()
                            prob_s[i_logit][sub_idx_2d] += prob

                        del logit_s, node

                # Combine all crops
                for i in range(4):
                    prob_s[i] = prob_s[i] / count_2d[:,:,None]
                prob_in = np.concatenate(prob_s, axis=-1)
                node_s = node_s / count_1d[:, None]

                # Clear cache memory
                torch.cuda.empty_cache()

                # Refinement
                node_s = torch.tensor(node_s).to(self.device).unsqueeze(0)
                seq = msa[:,0].to(self.device)
                idx_pdb = idx_pdb.to(self.device)
                prob_in = torch.tensor(prob_in).to(self.device).unsqueeze(0)

                with torch.cuda.amp.autocast():
                    xyz, lddt = self.model(node_s, seq, idx_pdb, prob_s=prob_in, refine_only=True)
            else:
                # Process full sequence
                print(f"Processing full sequence (L={L})")
                msa = msa[:,:1000].to(self.device)
                seq = msa[:,0]
                idx_pdb = idx_pdb.to(self.device)
                t1d = t1d[:,:10].to(self.device)
                t2d = t2d[:,:10].to(self.device)

                with torch.cuda.amp.autocast():
                    logit_s, _, xyz, lddt = self.model(msa, seq, idx_pdb, t1d=t1d, t2d=t2d)

                prob_s = list()
                for logit in logit_s:
                    prob = self.active_fn(logit.float())
                    prob = prob.reshape(-1, L, L).permute(1,2,0).cpu().numpy()
                    prob_s.append(prob)

            # Save predictions
            np.savez_compressed("%s.npz"%(out_prefix), 
                               dist=prob_s[0].astype(np.float16),
                               omega=prob_s[1].astype(np.float16),
                               theta=prob_s[2].astype(np.float16),
                               phi=prob_s[3].astype(np.float16))

            self.write_pdb(seq[0], xyz[0], idx_pdb[0], Bfacts=lddt[0], 
                          prefix="%s_init"%(out_prefix))

            # Run TRFold for final refinement
            prob_trF = list()
            for prob in prob_s:
                prob = torch.tensor(prob).permute(2,0,1).to(self.device)
                prob += 1e-8
                prob = prob / torch.sum(prob, dim=0)[None]
                prob_trF.append(prob)

            xyz = xyz[0, :, :, 1]
            TRF = TRFold(prob_trF, fold_params)
            with torch.set_grad_enabled(True):
                xyz = TRF.fold(xyz, batch=15, lr=0.1, nsteps=200)
            xyz = xyz.detach().cpu().numpy()

            # Add O and Cb
            N = xyz[:,0,:]
            CA = xyz[:,1,:]
            C = xyz[:,2,:]
            O = self.extend(np.roll(N, -1, axis=0), CA, C, 1.231, 2.108, -3.142)
            xyz = np.concatenate((xyz, O[:,None,:]), axis=1)

            self.write_pdb(seq[0], xyz, idx_pdb[0], Bfacts=lddt[0], prefix=out_prefix)

            print(f"Predictions saved to {out_prefix}.npz and {out_prefix}.pdb")

    def extend(self, a, b, c, L, A, D):
        '''
        input: 3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
        output: 4th coord
        '''
        N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
        bc = N(b-c)
        n = N(np.cross(b-a, bc))
        m = [bc, np.cross(n,bc), n]
        d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
        return c + sum([m*d for m,d in zip(m,d)])

    def write_pdb(self, seq, atoms, idx, Bfacts=None, prefix=None):
        L = len(seq)
        filename = "%s.pdb"%prefix
        ctr = 1
        with open(filename, 'wt') as f:
            if Bfacts == None:
                Bfacts = np.zeros(L)
            else:
                Bfacts = torch.clamp(Bfacts, 0, 1)

            for i,s in enumerate(seq):
                if (len(atoms.shape)==2):
                    f.write("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, " CA ", util.num2aa[s],
                        "A", idx[i]+1, atoms[i,0], atoms[i,1], atoms[i,2],
                        1.0, Bfacts[i]))
                    ctr += 1
                elif atoms.shape[1]==3:
                    for j,atm_j in enumerate((" N "," CA "," C ")):
                        f.write("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, atm_j, util.num2aa[s],
                            "A", idx[i]+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                            1.0, Bfacts[i]))
                        ctr += 1
                elif atoms.shape[1]==4:
                    for j,atm_j in enumerate((" N "," CA "," C ", " O ")):
                        f.write("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, atm_j, util.num2aa[s],
                            "A", idx[i]+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                            1.0, Bfacts[i]))
                        ctr += 1


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='RoseTTAFold prediction with custom c6d template features')

    parser.add_argument("-m", dest="model_dir", default="%s/weights"%(script_dir),
                       help="Path to pre-trained network weights [%s/weights]"%script_dir)
    parser.add_argument("-i", dest="a3m_fn", required=True,
                       help="Input multiple sequence alignment (in a3m format). Use minimal MSA (single sequence) for template-only prediction.")
    parser.add_argument("-o", dest="out_prefix", required=True,
                       help="Prefix for output files. Outputs: [out_prefix].npz and [out_prefix].pdb")
    parser.add_argument("--c6d", dest="c6d_fn", default=None,
                       help="Path to c6d features file (numpy array with shape: n_templates, L, L, 4)")
    parser.add_argument("--confidence", dest="confidence", type=float, default=0.7,
                       help="Confidence score for t0d and t1d template features (default: 0.7)")
    parser.add_argument("--cpu", dest='use_cpu', default=False, action='store_true',
                       help="Use CPU instead of GPU")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists("%s.npz"%args.out_prefix):
        pred = Predictor(model_dir=args.model_dir, use_cpu=args.use_cpu)
        pred.predict(args.a3m_fn, args.out_prefix, 
                    c6d_fn=args.c6d_fn,
                    t0d_confidence=args.confidence)
    else:
        print(f"Output {args.out_prefix}.npz already exists, skipping prediction")


# USAGE EXAMPLES:
# ---------------
# 
# 1. With c6d templates and minimal MSA (template-focused prediction):
#    python predict_e2e_modified.py -i minimal_query.a3m -o output --c6d my_c6d.npy --confidence 0.7
#
# 2. With c6d templates and high confidence:
#    python predict_e2e_modified.py -i minimal_query.a3m -o output --c6d my_c6d.npy --confidence 0.8
#
# 3. Without templates (MSA-only prediction):
#    python predict_e2e_modified.py -i full_msa.a3m -o output
#
# NOTES:
# ------
# - For template-only reconstruction, create a minimal a3m file with just the query sequence
# - c6d file should be a numpy array with shape (n_templates, L, L, 4)
# - The c6d_to_t2d function needs to be implemented based on your specific c6d format
# - Default confidence of 0.7 provides strong template guidance while allowing MSA input
# - Sequence length L in c6d must match the sequence length in the a3m file

# Usage Example
# python predict_custom.py \
#     -i minimal_query.a3m \
#     -o reconstruction_test \
#     --c6d my_pdb_derived_c6d.npy \
#     --confidence 0.7




