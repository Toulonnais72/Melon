import streamlit as st
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem import rdFingerprintGenerator as rFG
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from fpdf import FPDF
from datetime import datetime
import io
import os
import tempfile
import gc
from matplotlib import pyplot as plt
import pubchempy as pcp


PUBCHEM_LOOKUP_BATCH_THRESHOLD = 50  # Only fetch names from PubChem if <= this many molecules

# -----------------
st.set_page_config(page_title="Molecule Activity Predictor - C.MARTIN 2025", layout="wide")

# --- Utilitaires ---
def safe_str(val):
    if val is None: return ""
    if isinstance(val, float) and np.isnan(val): return ""
    return str(val)

def sanitize_filename(s):
    return (
        str(s)
        .replace(':', '_').replace(' ', '_')
        .replace('/', '_').replace('\\', '_')
        .replace('*', '_').replace('?', '_')
        .replace('"', '_').replace('<', '_')
        .replace('>', '_').replace('|', '_')
    )

import pubchempy as pcp

def get_pubchem_name_from_smiles(smiles):
    """
    Essaie d'obtenir le nom PubChem, le nom IUPAC, ou le 1er synonyme, puis le CID sinon.
    """
    try:
        results = pcp.get_compounds(smiles, 'smiles')
        if results:
            if results[0].iupac_name:
                return results[0].iupac_name
            if results[0].synonyms and len(results[0].synonyms) > 0:
                return results[0].synonyms[0]
            # Tente le nom courant (title)
            if hasattr(results[0], "title") and results[0].title:
                return results[0].title
            # Sinon, le CID PubChem
            if hasattr(results[0], "cid"):
                return f"PubChem CID {results[0].cid}"
    except Exception as e:
        pass
    return smiles  # Fallback si rien trouvé

def get_pubchem_name_safe(smiles, timeout=2.0):
    try:
        results = pcp.get_compounds(smiles, 'smiles', as_dataframe=False)
        if results:
            if results[0].iupac_name:
                return results[0].iupac_name
            elif results[0].synonyms and len(results[0].synonyms) > 0:
                return results[0].synonyms[0]
    except Exception as e:
        pass
    return smiles

# --- SESSION STATE ---
if "chembl_targets" not in st.session_state: st.session_state.chembl_targets = []
if "target_chembl_id" not in st.session_state: st.session_state.target_chembl_id = None
if "desc_options" not in st.session_state:
    st.session_state.desc_options = {
        "Poids moléculaire": ("MolWt", Descriptors.MolWt),
        "LogP": ("LogP", Descriptors.MolLogP),
        "TPSA": ("TPSA", Descriptors.TPSA),
        "H Acceptors": ("NumHAcceptors", Descriptors.NumHAcceptors),
        "H Donors": ("NumHDonors", Descriptors.NumHDonors),
        "Cycles": ("RingCount", Descriptors.RingCount),
        "FractionCSP3": ("FractionCSP3", Descriptors.FractionCSP3),
        "Rotatable Bonds": ("NumRotatableBonds", Descriptors.NumRotatableBonds),
        "Heavy Atoms": ("HeavyAtomCount", Descriptors.HeavyAtomCount)
    }
if "last_selected_desc" not in st.session_state: st.session_state.last_selected_desc = ["Poids moléculaire", "LogP", "TPSA"]
if "clf" not in st.session_state: st.session_state.clf = None
if "df" not in st.session_state: st.session_state.df = None
if "session_entries" not in st.session_state: st.session_state.session_entries = []

desc_options = st.session_state.desc_options

# --- Menu latéral ---
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Menu",
    [
        "1. Import & Cible",
        "2. Sélection descripteurs & entraînement",
        "3. Tester une molécule",
        "4. Tester un fichier",
        "5. Historique & Export",
        "6. Rapport PDF"
    ]
)

# --- 1. Import & choix cible ---
if menu == "1. Import & Cible":
    st.header("Importer une liste de cibles ChEMBL (CSV)")
    target_file = st.file_uploader("Fichier de cibles (CSV, TSV, TXT)", type=["csv", "tsv", "txt"])
    if target_file:
        try:
            df_targets = pd.read_csv(target_file, sep=None, engine='python')
            st.session_state.chembl_targets = df_targets.to_dict(orient="records")
            st.success(f"{len(df_targets)} cibles importées.")
            st.dataframe(df_targets.head())
        except Exception as e:
            st.error(f"Erreur d'import : {e}")

    if st.session_state.chembl_targets:
        st.write("Recherche dans la liste des cibles importées :")

        # --- Mapping automatique insensible à la casse/espaces ---
        raw_keys = list(st.session_state.chembl_targets[0].keys())
        norm = lambda x: x.lower().replace(" ", "").replace("_", "")
        column_map = {}
        for logical, possibles in {
            "id": ["chemblid", "chembl_id", "chembl id", "target_chembl_id", "targetchemblid"],
            "name": ["prefname", "name", "targetname"],
            "organism": ["organism"]
        }.items():
            found = None
            for rk in raw_keys:
                if norm(rk) in [norm(poss) for poss in possibles]:
                    found = rk
                    break
            column_map[logical] = found

        st.caption(
            f"Colonnes utilisées : ID = {column_map['id']}, Nom = {column_map['name']}, Organisme = {column_map['organism']}")

        query = st.text_input("Tapez un nom, un ChemBL ID ou un organisme (recherche insensible à la casse)")
        filtered = []
        if query:
            query_l = query.lower()
            for i, t in enumerate(st.session_state.chembl_targets):
                v_id = safe_str(t.get(column_map["id"], "")).lower() if column_map["id"] else ""
                v_name = safe_str(t.get(column_map["name"], "")).lower() if column_map["name"] else ""
                v_org = safe_str(t.get(column_map["organism"], "")).lower() if column_map["organism"] else ""
                if query_l in v_id or query_l in v_name or query_l in v_org:
                    label = f"{safe_str(t.get(column_map['id'], ''))} | {safe_str(t.get(column_map['name'], ''))[:40]} | {safe_str(t.get(column_map['organism'], ''))[:22]}"
                    filtered.append((i, label))
            filtered = filtered[:30]
        else:
            # Par défaut : les 30 premiers
            for i, t in enumerate(st.session_state.chembl_targets[:30]):
                label = f"{safe_str(t.get(column_map['id'], ''))} | {safe_str(t.get(column_map['name'], ''))[:40]} | {safe_str(t.get(column_map['organism'], ''))[:22]}"
                filtered.append((i, label))

        if filtered:
            idx = st.selectbox(
                "Résultats filtrés (max 30)",
                options=[x[0] for x in filtered],
                format_func=lambda i: filtered[[x[0] for x in filtered].index(i)][1])
            sel_target = st.session_state.chembl_targets[idx]
            st.session_state.target_chembl_id = sel_target.get(column_map["id"])
            st.info(f"Cible sélectionnée : {st.session_state.target_chembl_id}")
        else:
            st.info("Aucun résultat pour cette recherche.")

    if st.button("Exemple (protéines humaines)"):
        example_targets = new_client.target.filter(target_type="SINGLE PROTEIN", organism="Homo sapiens")[:20]
        st.session_state.chembl_targets = example_targets
        st.experimental_rerun()

# --- 2. Sélection descripteurs et entraînement modèle ---
if menu == "2. Sélection descripteurs & entraînement":
    st.header("Sélection des descripteurs & entraînement")
    selected_desc = st.multiselect(
        "Descripteurs chimiques à utiliser",
        list(desc_options.keys()),
        default=st.session_state.last_selected_desc
    )
    st.session_state.last_selected_desc = selected_desc

    if st.session_state.target_chembl_id is None:
        st.warning("Veuillez d'abord sélectionner une cible.")
    else:
        if st.button("Lancer l'entraînement du modèle"):
            with st.spinner("Téléchargement et entraînement en cours…"):
                try:
                    res = new_client.activity.filter(
                        target_chembl_id=st.session_state.target_chembl_id,
                        standard_type="IC50"
                    )
                    mols, labels = [], []
                    for entry in res:
                        if entry.get("standard_value") and entry.get("canonical_smiles"):
                            try:
                                value = float(entry["standard_value"])
                                if value > 0:
                                    mols.append(entry["canonical_smiles"])
                                    labels.append(1 if value < 1000 else 0)
                            except Exception:
                                continue
                    df = pd.DataFrame({"smiles": mols, "active": labels})
                    df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
                    df = df[df["mol"].notnull()].reset_index(drop=True)

                    selected = st.session_state.last_selected_desc
                    for desc in selected:
                        name, func = desc_options[desc]
                        df[name] = df["mol"].apply(func)

                    # --- PATCH RDKit : MorganGenerator ---
                    mg = rFG.GetMorganGenerator(radius=2, fpSize=1024)
                    df["fp"] = df["mol"].apply(lambda m: mg.GetFingerprintAsNumPy(m).tolist())

                    X = [list(d) + f for d, f in zip(df[[desc_options[d][0] for d in selected]].values, df["fp"])]
                    y = df["active"].values

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    try:
                        clf_ = XGBClassifier(
                            n_estimators=100,
                            eval_metric='logloss',
                            tree_method='hist'
                        )
                        clf_.fit(X_train, y_train)
                        gpu_mode = False
                    except Exception:
                        st.info("GPU non disponible ou erreur GPU, entraînement sur CPU.")
                        clf_ = XGBClassifier(
                            n_estimators=100,
                            eval_metric='logloss',
                            tree_method='hist'
                        )
                        clf_.fit(X_train, y_train)
                        gpu_mode = False
                    acc = clf_.score(X_test, y_test)
                    st.session_state.clf = clf_
                    st.session_state.df = df
                    st.success(
                        f"Modèle entraîné sur {len(df)} molécules | Accuracy: {acc:.2f} | GPU: {'Oui' if gpu_mode else 'Non'}")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement : {e}")

# --- 3. Tester une molécule ---
if menu == "3. Tester une molécule":
    st.header("Tester une molécule (SMILES ou nom)")
    if st.session_state.clf is None:
        st.warning("Veuillez d'abord entraîner le modèle.")
    else:
        input_smiles = st.text_input("SMILES ou nom de molécule", "")
        if st.button("Prédire"):
            with st.spinner("Recherche et calcul…"):
                try:
                    mol_query = new_client.molecule.search(input_smiles)
                    user_smiles = mol_query[0].get("molecule_structures", {}).get("canonical_smiles") if mol_query else input_smiles
                    user_mol = Chem.MolFromSmiles(user_smiles)
                    if user_mol is None:
                        st.error("Structure SMILES invalide.")
                    else:
                        # 1. Essaye ChEMBL (ou nom trouvé par ChEMBL)
                        chem_name = ""
                        if mol_query:
                            chem_name = mol_query[0].get("pref_name") or mol_query[0].get("molecule_name", "")
                        # 2. Essaye _Name RDKit
                        if not chem_name and user_mol.HasProp("_Name"):
                            chem_name = user_mol.GetProp("_Name")
                        # 3. Essaye PubChem si nom non trouvé ou redondant
                        if (not chem_name or chem_name == user_smiles or chem_name == input_smiles):
                            chem_name = get_pubchem_name_from_smiles(user_smiles)
                        # 4. Fallback sur le SMILES si tout échoue
                        if not chem_name:
                            chem_name = user_smiles

                        selected = st.session_state.last_selected_desc
                        desc = [desc_options[d][1](user_mol) for d in selected]
                        # Utilise le même MorganGenerator que pour l'entraînement
                        mg = rFG.GetMorganGenerator(radius=2, fpSize=1024)
                        fp = mg.GetFingerprintAsNumPy(user_mol).tolist()
                        features = [desc + fp]
                        pred = st.session_state.clf.predict(features)[0]
                        proba = st.session_state.clf.predict_proba(features)[0][1]
                        img = Draw.MolToImage(user_mol, size=(300, 300))
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        buf.seek(0)
                        st.image(buf, caption="Structure moléculaire", width=250)
                        st.markdown(f"**Nom :** {chem_name}")
                        st.markdown(f"**SMILES :** `{user_smiles}`")
                        st.markdown(f"**Prédiction :** {'🟩 Actif' if pred else '🟥 Inactif'}")
                        st.markdown(f"**Probabilité :** `{proba:.2f}`")
                        # Enregistrement dans l'historique de session
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        safe_timestamp = sanitize_filename(timestamp)
                        if "session_entries" not in st.session_state:
                            st.session_state.session_entries = []
                        st.session_state.session_entries.append({
                            "name": chem_name,  # <-- le nom correct
                            "user_smiles": user_smiles,
                            "prediction": pred,
                            "proba": proba,
                            "desc": desc,
                            "fp": fp,
                            "timestamp": safe_timestamp,
                            "desc_names": [desc_options[d][0] for d in selected]
                        })
                except Exception as e:
                    st.error(f"Erreur : {e}")


# --- 4. Tester un fichier de molécules ---
if menu == "4. Tester un fichier":
    st.header("Tester un fichier de molécules (SDF, CSV, SMI, TXT)")
    if st.session_state.clf is None:
        st.warning("Veuillez d'abord entraîner le modèle.")
    else:
        file_mol = st.file_uploader("Choisissez un fichier de molécules", type=["sdf", "csv", "smi", "txt"])
        if file_mol:
            ext = os.path.splitext(file_mol.name)[-1].lower()
            smiles_list, names_list = [], []
            st.info("Lecture du fichier…")
            # --- Lecture fichier ---
            if ext == ".sdf":
                tmp_sdf = tempfile.NamedTemporaryFile(delete=False, suffix=".sdf")
                tmp_sdf.write(file_mol.read())
                tmp_sdf.close()
                suppl = Chem.SDMolSupplier(tmp_sdf.name)
                for mol in suppl:
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol)
                        name = ""
                        if mol.HasProp("_Name"):
                            name = mol.GetProp("_Name")
                        smiles_list.append(smiles)
                        names_list.append(name)
                del suppl
                gc.collect()
                try:
                    os.unlink(tmp_sdf.name)
                except PermissionError:
                    pass
            elif ext == ".csv":
                df_in = pd.read_csv(file_mol)
                if "smiles" in df_in.columns:
                    smiles_list = df_in["smiles"].tolist()
                else:
                    smiles_list = df_in.iloc[:, 0].tolist()
                if "name" in df_in.columns:
                    names_list = df_in["name"].astype(str).tolist()
                else:
                    names_list = ["" for _ in smiles_list]
            else:  # .smi ou .txt
                content = file_mol.read().decode("utf-8").splitlines()
                for line in content:
                    if line.strip():
                        parts = line.strip().split()
                        smiles_list.append(parts[0])
                        names_list.append(parts[1] if len(parts) > 1 else "")

            # --- Batch size / PubChem lookup ---
            batch_size = len(smiles_list)
            do_pubchem = True  # Pour forcer la récupération du nom PubChem même si >50

            results = []
            probas_for_hist = []
            pbar = st.progress(0, "Prédiction en cours…")
            mg = rFG.GetMorganGenerator(radius=2, fpSize=1024)
            for idx, smi in enumerate(smiles_list):
                try:
                    user_mol = Chem.MolFromSmiles(smi)
                    if (user_mol is None) or (not smi):
                        pred = "Erreur structure"
                        proba = float("nan")
                        desc = [float("nan")] * len(st.session_state.last_selected_desc)
                        fp = [0] * 1024
                        chem_name = ""
                    else:
                        desc = [desc_options[d][1](user_mol) for d in st.session_state.last_selected_desc]
                        fp = mg.GetFingerprintAsNumPy(user_mol).tolist()
                        features = [desc + fp]
                        pred = st.session_state.clf.predict(features)[0]
                        proba = st.session_state.clf.predict_proba(features)[0][1]
                        probas_for_hist.append(proba)
                        # Détermination du nom chimique
                        # PATCH: Ignore "name" si juste un chiffre, vide ou identique au SMILES
                        chem_name = names_list[idx].strip() if names_list and names_list[idx] else ""
                        if chem_name.isdigit() or chem_name == smi or chem_name == "" or chem_name.lower() == "nan":
                            chem_name = ""
                        if not chem_name and user_mol.HasProp("_Name"):
                            chem_name = user_mol.GetProp("_Name")
                        if (not chem_name or chem_name == smi) and do_pubchem:
                            chem_name = get_pubchem_name_safe(smi)
                        if not chem_name:
                            chem_name = smi

                    # Ajout colonne label humain
                    pred_label = "Actif" if pred == 1 or pred == "Actif" else ("Inactif" if pred == 0 or pred == "Inactif" else pred)
                    # Construction du row résultat
                    result_row = {
                        "name": chem_name,
                        "smiles": smi,
                        "prediction": pred,
                        "PredictionLabel": pred_label,
                        "proba": proba if not np.isnan(proba) else "",
                        **{desc_options[d][0]: v for d, v in zip(st.session_state.last_selected_desc, desc)}
                    }
                    results.append(result_row)

                    # --- Ajout à l'historique session ---
                    if pred not in ["Erreur", "Erreur structure"]:
                        entry_for_history = {
                            "name": chem_name,
                            "user_smiles": smi,
                            "prediction": pred,
                            "proba": proba if not np.isnan(proba) else "",
                            "desc": [result_row.get(desc_options[d][0], float("nan")) for d in
                                     st.session_state.last_selected_desc],
                            "fp": fp if 'fp' in locals() else [0] * 1024,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "desc_names": [desc_options[d][0] for d in st.session_state.last_selected_desc]
                        }
                        st.session_state.session_entries.append(entry_for_history)
                except Exception:
                    chem_name = names_list[idx] if names_list else smi
                    result_row = {
                        "name": chem_name,
                        "smiles": smi,
                        "prediction": "Erreur",
                        "PredictionLabel": "Erreur",
                        "proba": "",
                        **{desc_options[d][0]: float("nan") for d in st.session_state.last_selected_desc}
                    }
                    results.append(result_row)
                pbar.progress((idx + 1) / batch_size)
            pbar.empty()

            # --- Affichage tableau avec arrondi et label ---
            df_out = pd.DataFrame(results)
            display_cols = ["name", "smiles", "prediction", "PredictionLabel", "proba"] + [desc_options[d][0] for d in st.session_state.last_selected_desc]
            if not df_out.empty:
                for col in ["proba"] + [desc_options[d][0] for d in st.session_state.last_selected_desc]:
                    if col in df_out.columns:
                        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").round(3)
                st.dataframe(df_out[display_cols])

            # --- Histogramme des probabilités
            if probas_for_hist:
                st.subheader("Distribution des probabilités prédites (proba d'activité)")
                st.pyplot(plt.figure(figsize=(5, 3)))
                plt.hist(probas_for_hist, bins=10, color="royalblue", alpha=0.7)
                plt.xlabel("Probabilité d'être Actif")
                plt.ylabel("Nombre de molécules")
                plt.title("Histogramme des probabilités de prédiction")
                plt.grid(True)
                st.pyplot(plt.gcf())
                plt.close()

            # --- Note explicative
            st.markdown(
                "> **Légende** : \n"
                "`prediction` = 1/Actif si IC50 < 1000 nM, sinon 0/Inactif.\n\n"
                "`proba` = probabilité calculée par le modèle d'être actif.\n\n"
                "`MolWt`, `LogP`, etc. = descripteurs chimiques calculés via RDKit.\n\n"
                "`PredictionLabel` = version texte de la prédiction binaire."
            )

            # --- Export CSV
            csv = df_out[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger les résultats (CSV)", csv, "resultats_predictions.csv", "text/csv")
            gc.collect()


# --- 5. Historique & Export ---
if menu == "5. Historique & Export":
    st.header("Historique de session")
    entries = st.session_state.session_entries
    if not entries:
        st.info("Aucune molécule testée dans cette session.")
    else:
        df_hist = pd.DataFrame([
            {
                "Nom": e.get("name", ""),
                "SMILES": e.get("user_smiles", "") or e.get("smiles", ""),
                "Prédiction": "Actif" if str(e.get("prediction", "")).lower() in ("1", "actif") else "Inactif",
                "Probabilité": e.get("proba", "")
            } for e in entries
        ])
        st.dataframe(df_hist)
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("Exporter l'historique en CSV", csv, "historique_session.csv", "text/csv")

# --- 6. Rapport PDF ---
if menu == "6. Rapport PDF":
    st.header("Générer un rapport PDF global de la session")
    if not st.session_state.session_entries:
        st.info("Aucune molécule testée pour générer un rapport.")
    else:
        import matplotlib.pyplot as plt
        import seaborn as sns
        if st.button("Générer le rapport PDF"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Rapport de session - Prédictions moléculaires", ln=True, align="C")
            for entry in st.session_state.session_entries:
                pdf.add_page()
                # -- Patch pour toujours afficher le nom --
                nom = entry.get('name', '') or entry.get('compound_input', '')
                smiles = entry.get('user_smiles', '') or entry.get('smiles', '')
                pdf.cell(200, 10, txt=f"Molécule : {nom}", ln=True)
                pdf.cell(200, 10, txt=f"SMILES : {smiles}", ln=True)
                pred = entry.get('prediction', '')
                pred_txt = "Actif" if str(pred).lower() in ("1", "actif") else "Inactif"
                pdf.cell(200, 10, txt=f"Prédiction : {pred_txt}", ln=True)
                try:
                    proba_val = float(entry.get('proba', ''))
                    pdf.cell(200, 10, txt=f"Probabilité : {proba_val:.2f}", ln=True)
                except:
                    pdf.cell(200, 10, txt=f"Probabilité : {entry.get('proba', '')}", ln=True)
                # Affichage des descripteurs utilisés
                if "desc_names" in entry and "desc" in entry:
                    for i, d in enumerate(entry["desc_names"]):
                        try:
                            val = float(entry['desc'][i])
                            pdf.cell(200, 10, txt=f"{d} : {val:.2f}", ln=True)
                        except:
                            pdf.cell(200, 10, txt=f"{d} : {entry['desc'][i]}", ln=True)
                # Heatmap fingerprint
                if "fp" in entry:
                    import numpy as np
                    heatmap = np.array(entry['fp']).reshape(32, 32)
                    plt.figure(figsize=(4, 4))
                    sns.heatmap(heatmap, cbar=False, xticklabels=False, yticklabels=False)
                    plt.axis('off')
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        plt.savefig(tmp_img.name, bbox_inches='tight')
                        plt.close()
                        tmp_img_path = tmp_img.name
                    pdf.image(tmp_img_path, x=30, y=120, w=120)
                    try:
                        os.remove(tmp_img_path)
                    except PermissionError:
                        pass
                # Image molécule
                if smiles:
                    try:
                        from rdkit import Chem
                        from rdkit.Chem import Draw
                        mol = Chem.MolFromSmiles(smiles)
                        img = Draw.MolToImage(mol, size=(200, 200))
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_mol:
                            img.save(tmp_mol.name)
                            tmp_mol_path = tmp_mol.name
                        pdf.image(tmp_mol_path, x=60, y=40, w=80)
                        try:
                            os.remove(tmp_mol_path)
                        except PermissionError:
                            pass
                    except Exception:
                        pass

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as ftmp:
                pdf.output(ftmp.name)
                with open(ftmp.name, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="Télécharger le rapport PDF",
                    data=pdf_bytes,
                    file_name="rapport_session_prediction.pdf",
                    mime="application/pdf"
                )
                try:
                    os.remove(ftmp.name)
                except PermissionError:
                    pass


# --- Fin ---
