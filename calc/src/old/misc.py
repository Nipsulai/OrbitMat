def compute_energy_weighted_dm(kpt_data, K_vectors, T_vectors=None, betas=_BETAS):
    """Compute R-space energy-weighted density matrices via Fourier transform.

    D(R) = (1/N_total) Σ_{k ∈ full BZ} D(k) exp(i·2π k·R)

    Only irreducible k-points are available; time-reversal gives D(-k) = D(k) for
    real MO coefficients (as printed in the MOLog). Each irreducible k contributes:
      - non-self-conjugate (k ≠ -k mod G): factor 2, two terms → 2·D(k)·cos(2π k·R)
      - self-conjugate (2k ∈ Z³, e.g. Γ):  factor 1, one term →   D(k)·cos(2π k·R)

    The result is purely real by construction (no imaginary part to discard).

    kpt_data keys are 1-based and map to K_vectors[key-1].
    K_vectors: fractional reciprocal lattice coordinates (dimensionless).
    T_vectors: integer lattice translation vectors. Defaults to [(0,0,0)].

    Returns: {beta: {'Dh': {T_tuple: ndarray[nao,nao]},
                     'Dp': {T_tuple: ndarray[nao,nao]}}} (float32)
    """
    if not kpt_data:
        return {}

    first = next(iter(kpt_data.values()))
    nao   = first['coefficients'].shape[0]

    if T_vectors is None:
        T_vectors = np.zeros((1, 3), dtype=int)
    T_vectors = np.asarray(T_vectors)

    # ── Step 1: compute D_h(k) and D_p(k) for each irreducible k-point ──────
    k_Dh = {}
    k_Dp = {}

    for kpt_idx, kdata in kpt_data.items():
        eps = kdata['eigenvalues_au']
        occ = kdata['occupations']
        C   = kdata['coefficients'].astype(np.float64)

        max_occ = occ.max()
        n = occ / max_occ if max_occ > 0 else occ.copy()

        occ_mask = n > 0.5
        if not np.any(occ_mask):
            continue
        eps_homo    = eps[occ_mask].max()
        unocc_mask  = ~occ_mask
        eps_lumo    = eps[unocc_mask].min() if np.any(unocc_mask) else eps_homo

        k_Dh[kpt_idx] = {}
        k_Dp[kpt_idx] = {}
        for beta in betas:
            h_arg = np.where(n > 0, np.clip(-beta * (eps_homo - eps), -700.0, 0.0), -700.0)
            p_arg = np.where(n < 1, np.clip( beta * (eps_lumo - eps), -700.0, 0.0), -700.0)
            k_Dh[kpt_idx][beta] = (C * np.exp(h_arg) * n        ) @ C.T
            k_Dp[kpt_idx][beta] = (C * np.exp(p_arg) * (1.0 - n)) @ C.T

    if not k_Dh:
        return {}

    # ── Step 2: classify k-points as self-conjugate (2k ∈ Z³) or not ────────
    # Self-conjugate k-points (e.g. Γ, zone-boundary) appear once in the full BZ;
    # all others appear as a k / -k pair.
    factors = {}
    for kpt_idx in k_Dh:
        two_k = 2.0 * K_vectors[kpt_idx - 1]
        is_self = np.allclose(two_k - np.round(two_k), 0.0, atol=1e-6)
        factors[kpt_idx] = 1 if is_self else 2

    n_self  = sum(1 for v in factors.values() if v == 1)
    N_total = 2 * len(k_Dh) - n_self  # total k-points in full BZ

    # ── Step 3: Fourier transform for every T-vector ─────────────────────────
    result = {b: {'Dh': {}, 'Dp': {}} for b in betas}

    for T_vec in T_vectors:
        T = tuple(int(x) for x in T_vec)

        Dh_R = {b: np.zeros((nao, nao)) for b in betas}
        Dp_R = {b: np.zeros((nao, nao)) for b in betas}

        for kpt_idx in k_Dh:
            k_vec  = K_vectors[kpt_idx - 1]
            # factor · cos(2π k·R)  combines the +k and -k contributions
            weight = factors[kpt_idx] * np.cos(2.0 * np.pi * float(np.dot(k_vec, T_vec)))

            for beta in betas:
                Dh_R[beta] += weight * k_Dh[kpt_idx][beta]
                Dp_R[beta] += weight * k_Dp[kpt_idx][beta]

        for beta in betas:
            result[beta]['Dh'][T] = (Dh_R[beta] / N_total).astype(np.float32)
            result[beta]['Dp'][T] = (Dp_R[beta] / N_total).astype(np.float32)

    return result



def get_topk_blocks(workdir, method, nao, atoms, T_vectors, norb_z, lattice_vecs,
    topk=32, cutoff=1e-32):
    """
    Extract top-k interaction blocks for each atom from periodic DFT matrices.
    """
    perm = METHODS[method].orbital_perm if method in METHODS else "xtb"
    if perm == "dz":
        permute_block = _permute_block_pbe
    elif perm == "scan":
        permute_block = _permute_block_scan
    elif perm == "tzvp":
        permute_block = _permute_block_tzvp
    else:
        permute_block = _permute_block

    n_atoms = len(atoms)

    blocks_by_source = defaultdict(list)
    atom_ranges = build_atom_ranges(norb_z, atoms)

    atom_coords = np.array([a['coord_bohr'] for a in atoms], dtype=np.float64)
    atom_numbers = np.array([a['atomnum'] for a in atoms], dtype=np.int32)

    a1, a2, a3 = lattice_vecs

    # Store
    T_matrix_cache = {}

    # Step 1: Score all blocks and cache matrices
    for T_idx, T_vec in enumerate(T_vectors):
        #T_norm = np.linalg.norm(T_vec)

        # Get matrices for the translation
        F_T, P_T, S_T, H_T = (
            parse_matrix(workdir, method, key, nao, binary=True, T_idx=T_idx, keep_sparse=True)
            for key in ("KS", "P", "S", "HCORE")
        )

        # Store matrices
        T_matrix_cache[T_idx] = {'KS': F_T, 'P': P_T, 'S': S_T, 'HCORE': H_T}

        # Extract atom-atom blocks
        for src in range(1, n_atoms + 1):
            for ngb in range(1, n_atoms + 1):
                cell = tuple(T_vec)

                # Compute norms
                nF = compute_block_norm_squared_sparse(F_T, src, ngb, atom_ranges)
                nP = compute_block_norm_squared_sparse(P_T, src, ngb, atom_ranges)
                nS = compute_block_norm_squared_sparse(S_T, src, ngb, atom_ranges)
                nH = compute_block_norm_squared_sparse(H_T, src, ngb, atom_ranges)
                score = float(nF * nS * nH)

                if score < cutoff:
                    continue

                # Store block metadata
                blocks_by_source[src].append({
                    'ngb': ngb,
                    'cell': cell,
                    'T_idx': T_idx,
                    'score': score,
                    'Fscr': nF, 'Pscr': nP, 'Sscr': nS, 'Hscr': nH
                })

    # Phase 2: Top-k selection and permutation
    topk_blocks: List[Dict[str, Any]] = []
    self_ctr, pair_ctr = 0, 0

    # Process each source atom
    for src in range(1, n_atoms + 1):
        atom_blocks = blocks_by_source.get(src, [])

        # Separate self-blocks from neighbor blocks
        self_block = None
        neighbor_blocks = []

        for block in atom_blocks:
            if block['ngb'] == src and block['cell'] == (0, 0, 0):
                self_block = block
            else:
                neighbor_blocks.append(block)

        # Add self-block if it exists
        if self_block is not None:
            T_idx = self_block['T_idx']
            ngb = self_block['ngb']
            z_src = atom_numbers[src - 1]
            z_ngb = atom_numbers[ngb - 1]
            F_block = get_block(T_matrix_cache[T_idx]["KS"], src, ngb, atom_ranges)
            P_block = get_block(T_matrix_cache[T_idx]["P"], src, ngb, atom_ranges)
            S_block = get_block(T_matrix_cache[T_idx]["S"], src, ngb, atom_ranges)
            H_block = get_block(T_matrix_cache[T_idx]["HCORE"], src, ngb, atom_ranges)

            topk_blocks.append({
                "is_self": True,
                "ctr": self_ctr,
                "source": src,
                "neighbor": ngb,
                "cell": self_block['cell'],
                "matrix": {
                    'F': permute_block(F_block, z_src, z_ngb), 'P': permute_block(P_block, z_src, z_ngb), 'S': permute_block(S_block, z_src, z_ngb), 'H': permute_block(H_block, z_src, z_ngb),
                    'score': self_block['score'],
                    'Fscr': self_block['Fscr'],
                    'Pscr': self_block['Pscr'],
                    'Sscr': self_block['Sscr'],
                    'Hscr': self_block['Hscr']
                },
            })
            self_ctr += 1

        # Sort neighbor blocks by score
        neighbor_blocks.sort(key=lambda b: b['score'], reverse=True)

        # Apply top-k with tie-breaking
        if len(neighbor_blocks) <= topk:
            selected_blocks = neighbor_blocks
        else:
            #kth_score = neighbor_blocks[topk - 1]['score']
            selected_blocks = sorted(neighbor_blocks, key=lambda b: b['score'], reverse=True)[:topk]
            #rtol = 1e-12
            #selected_blocks = [
            #    b for b in neighbor_blocks
            #   if b['score'] > kth_score or abs(b['score'] - kth_score) <= rtol * max(1.0, abs(kth_score))
            #]

        # Add selected blocks to output
        for block in selected_blocks:
            ngb = block['ngb']
            Rvec = block['cell']
            T_idx = block['T_idx']

            F_block = get_block(T_matrix_cache[T_idx]["KS"], src, ngb, atom_ranges)
            P_block = get_block(T_matrix_cache[T_idx]["P"], src, ngb, atom_ranges)
            S_block = get_block(T_matrix_cache[T_idx]["S"], src, ngb, atom_ranges)
            H_block = get_block(T_matrix_cache[T_idx]["HCORE"], src, ngb, atom_ranges)

            z_src = atom_numbers[src - 1]
            z_ngb = atom_numbers[ngb - 1]

            perm_mat = {
                'H': permute_block(H_block, z_src, z_ngb),
                'S': permute_block(S_block, z_src, z_ngb),
                'F': permute_block(F_block, z_src, z_ngb),
                'P': permute_block(P_block, z_src, z_ngb),
                'score': block['score'],
                'Fscr': block['Fscr'],
                'Pscr': block['Pscr'],
                'Sscr': block['Sscr'],
                'Hscr': block['Hscr']
            }

            # Calculate distance
            ic1, ic2, ic3 = Rvec
            r_i = atom_coords[src - 1]
            r_j = atom_coords[ngb - 1]
            shift = ic1 * a1 + ic2 * a2 + ic3 * a3
            dvec = r_j - r_i + shift
            dist = float(np.linalg.norm(dvec))

            topk_blocks.append({
                "is_self": False,
                "ctr": pair_ctr,
                "source": src,
                "neighbor": ngb,
                "cell": Rvec,
                "matrix": perm_mat,
                "dist": dist,
            })
            pair_ctr += 1

    return topk_blocks

def parse_molog_full(workdir):
    """Parse CP2K MOLog file including eigenvectors.

    Returns dict: {kpt_idx (1-based): {
        'eigenvalues_au': ndarray[nmo],
        'occupations':    ndarray[nmo],
        'coefficients':   ndarray[nao, nmo],  # real, float32
    }}
    Only the irreducible k-points are present (as printed by CP2K).
    """
    filepath = workdir / "eigenvalues-1_0.MOLog"

    with open(filepath) as fh:
        raw = fh.readlines()

    lines = [l.strip() for l in raw]

    def _is_int(x):
        try:
            int(x)
            return True
        except ValueError:
            return False

    # Find k-point section starts
    kpt_starts = []
    for i, s in enumerate(lines):
        m = re.match(r'MO\|\s+EIGENVALUES.*K POINT\s+(\d+)', s)
        if m:
            kpt_starts.append((i, int(m.group(1))))

    result = {}

    for sec_num, (start, kpt_idx) in enumerate(kpt_starts):
        end = kpt_starts[sec_num + 1][0] if sec_num + 1 < len(kpt_starts) else len(lines)
        sec = lines[start:end]

        eigenvalues = []
        occupations = []
        coeff_dict = {}  # mo_idx (0-based) -> {ao_idx (0-based): float}

        i = 1  # skip k-point header
        while i < len(sec):
            s = sec[i]
            if not s or s == 'MO|' or not s.startswith('MO|'):
                i += 1
                continue

            content = s[3:].strip()
            if not content:
                i += 1
                continue

            parts = content.split()

            # MO indices block: all tokens are integers (no decimal point)
            if all(_is_int(p) for p in parts):
                mo_indices = [int(p) for p in parts]
                i += 1

                # Next non-empty MO| line → eigenvalues
                while i < len(sec) and (not sec[i].startswith('MO|') or not sec[i][3:].strip()):
                    i += 1
                if i >= len(sec):
                    break
                eigenvalues.extend(float(v) for v in sec[i][3:].strip().split())
                i += 1

                # Skip blank MO| lines → occupation numbers
                while i < len(sec) and (not sec[i].startswith('MO|') or not sec[i][3:].strip()):
                    i += 1
                if i >= len(sec):
                    break
                occupations.extend(float(v) for v in sec[i][3:].strip().split())
                i += 1

                # Skip blank MO| lines → first coefficient row
                while i < len(sec) and (not sec[i].startswith('MO|') or not sec[i][3:].strip()):
                    i += 1

                # Coefficient rows: "ao_idx  atom_idx  elem  orbital  c1  c2  ..."
                while i < len(sec):
                    row = sec[i]
                    if not row.startswith('MO|'):
                        i += 1
                        break
                    rc = row[3:].strip()
                    if not rc:
                        i += 1
                        continue  # blank line between atom groups — keep reading
                    rp = rc.split()
                    if len(rp) < 5 or not _is_int(rp[0]) or not _is_int(rp[1]):
                        break
                    ao_idx = int(rp[0]) - 1
                    try:
                        coeffs = [float(v) for v in rp[4: 4 + len(mo_indices)]]
                    except ValueError:
                        i += 1
                        continue
                    for offset, coeff in enumerate(coeffs):
                        mo_idx = mo_indices[offset] - 1
                        coeff_dict.setdefault(mo_idx, {})[ao_idx] = coeff
                    i += 1
            else:
                i += 1

        if eigenvalues and coeff_dict:
            nmo = len(eigenvalues)
            nao = max(max(d.keys()) for d in coeff_dict.values()) + 1
            C = np.zeros((nao, nmo), dtype=np.float32)
            for mo_idx, ao_dict in coeff_dict.items():
                for ao_idx, coeff in ao_dict.items():
                    if mo_idx < nmo and ao_idx < nao:
                        C[ao_idx, mo_idx] = coeff
            result[kpt_idx] = {
                'eigenvalues_au': np.array(eigenvalues, dtype=np.float64),
                'occupations':    np.array(occupations, dtype=np.float64),
                'coefficients':   C,
            }

    return result