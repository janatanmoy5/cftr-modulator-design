#!/usr/bin/env python3
"""Local CFTR molecule-design and QSAR prediction web server.

Run:  .venv/bin/python app.py --open
Then: http://127.0.0.1:8765

Research prioritization only. Predictions are not clinical evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import re
import threading
import webbrowser
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"


def load_numbered(path: Path):
    spec = importlib.util.spec_from_file_location("cftr_features", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


FEATURES = load_numbered(ROOT / "src" / "09_featurize_chembl_compounds.py")
SVR = joblib.load(MODELS / "bioactivity_svr_rbf_descriptors_fp.joblib")
CLASSIFIER = joblib.load(MODELS / "cftr_activity_classifier.joblib")
REGRESSOR = joblib.load(MODELS / "cftr_potency_regressor.joblib")
WEB_DOCK = load_numbered(ROOT / "src" / "web_docking.py")
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
DOCK_JOBS, DOCK_LOCK = {}, threading.Lock()

TRAIN = pd.read_csv(PROCESSED / "chembl_bioactivity_features.csv",
                    usecols=["molecule_chembl_id", "canonical_smiles"])
TRAIN_FPS, TRAIN_IDS = [], []
for row in TRAIN.itertuples(index=False):
    mol = Chem.MolFromSmiles(str(row.canonical_smiles))
    if mol:
        TRAIN_FPS.append(MORGAN.GetFingerprint(mol)); TRAIN_IDS.append(row.molecule_chembl_id)


def model_frame(desc: dict, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{col: desc.get(col, np.nan) for col in columns}])


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "CFTR-Molecule-Designer/1.0"})
    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError("Compound identifier or chemical name was not found") from exc
        raise ValueError(f"Remote compound service returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError("Could not reach the ChEMBL/PubChem compound service") from exc


def resolve_compound(query: str) -> tuple[str, str]:
    value = query.strip()
    if not value:
        raise ValueError("Enter a SMILES, ChEMBL ID, PubChem CID, or chemical name")

    # Prefer a valid local structure so normal SMILES prediction remains offline.
    if Chem.MolFromSmiles(value) is not None:
        return value, "SMILES"

    if re.fullmatch(r"CHEMBL\d+", value, flags=re.IGNORECASE):
        chembl_id = value.upper()
        data = fetch_json(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json")
        smiles = (data.get("molecule_structures") or {}).get("canonical_smiles")
        if not smiles:
            raise ValueError(f"{chembl_id} has no canonical SMILES in ChEMBL")
        return smiles, chembl_id

    cid_match = re.fullmatch(r"(?:(?:PUBCHEM|CID)\s*:?\s*)?(\d+)", value, flags=re.IGNORECASE)
    if cid_match:
        cid = cid_match.group(1)
        endpoint = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES/JSON"
        source = f"PubChem CID {cid}"
    else:
        endpoint = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" +
                    quote(value, safe="") + "/property/SMILES/JSON")
        source = f"PubChem name: {value}"

    data = fetch_json(endpoint)
    properties = data.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        raise ValueError(f"No structure was returned for {value}")
    record = properties[0]
    smiles = next((record.get(key) for key in
                   ("SMILES", "ConnectivitySMILES", "CanonicalSMILES", "IsomericSMILES")
                   if record.get(key)), None)
    if not smiles:
        raise ValueError(f"No SMILES was returned for {value}")
    return smiles, source


def predict(query: str) -> dict:
    smiles, resolved_from = resolve_compound(query)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse this SMILES string")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    desc = FEATURES.descriptors_for_smiles(canonical)
    svr_px = float(SVR["pipeline"].predict(model_frame(desc, SVR["feature_cols"]))[0])
    active = float(CLASSIFIER["pipeline"].predict_proba(
        model_frame(desc, CLASSIFIER["feature_cols"]))[0, 1])
    integrated_px = float(REGRESSOR["pipeline"].predict(
        model_frame(desc, REGRESSOR["feature_cols"]))[0])

    fp = MORGAN.GetFingerprint(mol)
    similarities = DataStructs.BulkTanimotoSimilarity(fp, TRAIN_FPS)
    best_i = int(np.argmax(similarities)); max_sim = float(similarities[best_i])
    ad_label = "Inside" if max_sim >= 0.60 else ("Borderline" if max_sim >= 0.40 else "Outside")
    consensus = (svr_px + integrated_px) / 2
    return {
        "resolved_from": resolved_from,
        "canonical_smiles": canonical,
        "structure_svg": Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(430, 300),
                                                useSVG=True, legends=[canonical]),
        "svr_predicted_px": round(svr_px, 3),
        "svr_approx_nM": round(10 ** (9 - svr_px), 2),
        "integrated_predicted_px": round(integrated_px, 3),
        "consensus_predicted_px": round(consensus, 3),
        "predicted_active_probability": round(active, 4),
        "activity_class": "ACTIVE" if active >= 0.5 else "INACTIVE",
        "activity_call": ("Predicted active at probability threshold 0.50" if active >= 0.5
                          else "Predicted inactive at probability threshold 0.50"),
        "applicability_domain": ad_label,
        "nearest_training_similarity": round(max_sim, 3),
        "nearest_training_compound": str(TRAIN_IDS[best_i]),
        "properties": {
            "Molecular weight": round(desc["mol_weight"], 2),
            "cLogP": round(desc["logp"], 2), "TPSA": round(desc["tpsa"], 2),
            "H-bond donors": int(desc["h_bond_donors"]),
            "H-bond acceptors": int(desc["h_bond_acceptors"]),
            "Rotatable bonds": int(desc["rotatable_bonds"]),
            "QED drug-likeness": round(desc["qed_druglikeness"], 3),
            "Lipinski violations": int(desc["lipinski_violations"]),
        },
        "docking": "Not run in instant mode; use screen_library.sh for CFTR pocket docking.",
        "warning": "Computational research prediction only; not evidence of CFTR modulation or clinical benefit.",
    }


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CFTR Molecule Designer</title>
<script type="text/javascript" src="https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"></script>
<style>
:root{--ink:#10231d;--muted:#63736d;--teal:#087f6d;--lime:#d9f263;--paper:#f3f6ef;--card:#fff;--line:#dce5dd;--warn:#8b5d00}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
header{padding:28px clamp(18px,5vw,72px);background:#092b25;color:white;display:flex;justify-content:space-between;gap:20px;align-items:end}
h1{margin:0;font-size:clamp(27px,4vw,48px);letter-spacing:-.04em}header p{margin:4px 0 0;color:#bcd0c9}.badge{background:var(--lime);color:#18372c;padding:7px 11px;border-radius:99px;font-weight:800;white-space:nowrap}
main{max-width:1400px;margin:auto;padding:24px;display:grid;grid-template-columns:minmax(430px,1.05fr) minmax(380px,.95fr);gap:22px}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 8px 28px #163c2c0d}h2{margin:0 0 14px;font-size:20px}.editor{min-height:420px;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:white}label{font-weight:750;display:block;margin:16px 0 6px}textarea{width:100%;min-height:74px;padding:12px;border:1px solid #aebdb5;border-radius:10px;font:14px ui-monospace,monospace;resize:vertical}.actions{display:flex;gap:9px;margin-top:12px;flex-wrap:wrap}button{border:0;border-radius:10px;padding:11px 17px;font-weight:800;cursor:pointer;background:var(--teal);color:white}button.secondary{background:#e5ece7;color:var(--ink)}button:disabled{opacity:.55}.status{min-height:24px;color:var(--muted);margin-top:10px}.hero-results{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.metric{background:#eef4ef;border-radius:12px;padding:13px}.metric strong{display:block;font-size:24px}.metric span{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.metric.primary{background:#dff3ec}.metric.accent{background:#eff8c7}.structure{display:grid;place-items:center;min-height:280px;border-bottom:1px solid var(--line);margin-bottom:15px;overflow:auto}.structure svg{max-width:100%;height:auto}.props{display:grid;grid-template-columns:1fr 1fr;gap:7px 18px}.prop{display:flex;justify-content:space-between;border-bottom:1px dotted #ccd7cf;padding:5px 0}.notice{margin-top:14px;padding:11px;border-left:4px solid #d69b19;background:#fff8de;color:#624600}.hidden{display:none}.ad-Inside{color:#087f2b}.ad-Borderline{color:#a56600}.ad-Outside{color:#b3261e}.dockbox{margin-top:15px;padding:14px;border:1px solid var(--line);border-radius:12px}.dockgrid{display:grid;grid-template-columns:1fr auto;gap:6px 14px}.dockimg{width:100%;margin-top:12px;border-radius:10px;background:#000}.downloads a{display:inline-block;margin:7px 10px 0 0;color:var(--teal);font-weight:800}
@media(max-width:920px){main{grid-template-columns:1fr}.editor{min-height:360px}}@media(max-width:560px){main{padding:12px}.card{padding:14px}.hero-results{grid-template-columns:1fr}.props{grid-template-columns:1fr}header{align-items:start;flex-direction:column}}
</style></head><body><header><div><h1>CFTR Molecule Designer</h1><p>Sketch chemistry. Estimate CFTR activity and potency. Check model applicability.</p></div><div class="badge">Research use only</div></header>
<main><section class="card"><h2>1. Design or find a molecule</h2><div id="jsme" class="editor"></div><label for="smiles">SMILES, ChEMBL ID, PubChem CID, or chemical name</label><textarea id="smiles" placeholder="Examples: c1ccccc1, CHEMBL25, CID 2244, or aspirin"></textarea><div class="actions"><button id="predict">Predict CFTR profile</button><button class="secondary" id="example">Load example</button><button class="secondary" id="clear">Clear</button></div><div class="status" id="status">Editor loading…</div></section>
<section class="card"><h2>2. CFTR prediction</h2><div id="empty"><p>Submit a valid molecule to generate its profile.</p></div><div id="results" class="hidden"><div id="structure" class="structure"></div><div class="hero-results"><div class="metric primary"><span>RBF-SVR pX</span><strong id="svr"></strong><small id="nm"></small></div><div class="metric accent"><span>CFTR class</span><strong id="classcall"></strong><small id="prob"></small><br><small id="call"></small></div><div class="metric"><span>Integrated pX</span><strong id="rf"></strong></div><div class="metric"><span>Consensus pX</span><strong id="consensus"></strong></div></div><h2 style="margin-top:20px">Applicability & properties</h2><p>Domain: <strong id="ad"></strong> · nearest training compound: <strong id="nearest"></strong></p><div id="props" class="props"></div><div class="dockbox"><h2>3. Structural docking</h2><button id="dock">Run docking against 5 CFTR pockets</button><div class="status" id="dockstatus">Longer calculation; starts only when requested.</div><div id="dockresults" class="hidden"><div class="dockgrid" id="dockgrid"></div><div class="downloads" id="downloads"></div><img id="dockimg" class="dockimg hidden" alt="Best docked CFTR receptor–ligand complex"></div></div><div class="notice" id="warning"></div></div></section></main>
<script>
let jsmeApplet=null,currentSmiles=''; function jsmeOnLoad(){jsmeApplet=new JSApplet.JSME("jsme","100%","420px",{options:"newlook,star"});document.getElementById('status').textContent="Editor ready";jsmeApplet.setAfterStructureModifiedCallback(()=>document.getElementById('smiles').value=jsmeApplet.smiles())}
const $=id=>document.getElementById(id); $('example').onclick=()=>{const s='CC(=O)OC1=CC=CC=C1C(=O)O';$('smiles').value=s;if(jsmeApplet)jsmeApplet.readGenericMolecularInput(s)};$('clear').onclick=()=>{$('smiles').value='';if(jsmeApplet)jsmeApplet.reset();$('results').classList.add('hidden');$('empty').classList.remove('hidden')}
$('predict').onclick=async()=>{let smiles=$('smiles').value.trim()||(jsmeApplet?jsmeApplet.smiles():'');if(!smiles){$('status').textContent='Draw a molecule or enter a SMILES, ChEMBL ID, PubChem CID, or chemical name.';return}$('predict').disabled=true;$('status').textContent='Resolving structure and calculating model predictions…';try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles})});const d=await r.json();if(!r.ok)throw Error(d.error||'Prediction failed');currentSmiles=d.canonical_smiles;$('empty').classList.add('hidden');$('results').classList.remove('hidden');$('structure').innerHTML=d.structure_svg;$('svr').textContent=d.svr_predicted_px;$('nm').textContent='≈ '+d.svr_approx_nM+' nM';$('classcall').textContent=d.activity_class;$('prob').textContent=(100*d.predicted_active_probability).toFixed(1)+'% probability';$('call').textContent=d.activity_call;$('rf').textContent=d.integrated_predicted_px;$('consensus').textContent=d.consensus_predicted_px;$('ad').textContent=d.applicability_domain;$('ad').className='ad-'+d.applicability_domain;$('nearest').textContent=d.nearest_training_compound+' (Tanimoto '+d.nearest_training_similarity+')';$('props').innerHTML=Object.entries(d.properties).map(([k,v])=>`<div class="prop"><span>${k}</span><strong>${v}</strong></div>`).join('');$('warning').textContent=d.warning;$('status').textContent='Prediction complete · resolved from '+d.resolved_from+' · '+d.canonical_smiles}catch(e){$('status').textContent=e.message}finally{$('predict').disabled=false}}
$('dock').onclick=async()=>{if(!currentSmiles){$('dockstatus').textContent='Run the instant prediction first.';return}$('dock').disabled=true;$('dockresults').classList.add('hidden');$('dockstatus').textContent='Starting docking job…';try{let r=await fetch('/api/dock',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles:currentSmiles})});let d=await r.json();if(!r.ok)throw Error(d.error||'Could not start docking');while(true){await new Promise(x=>setTimeout(x,2000));r=await fetch('/api/dock/'+d.job_id);d=await r.json();$('dockstatus').textContent=d.message||d.status;if(d.status==='failed')throw Error(d.error);if(d.status==='complete')break}$('dockresults').classList.remove('hidden');$('dockgrid').innerHTML=d.result.scores.map(x=>`<span>${x.binding_site}</span><strong>${Number(x.best_affinity_kcal_mol).toFixed(3)} kcal/mol</strong>`).join('');$('downloads').innerHTML=`<a href="/artifact/${d.result.best_pose_pdbqt}" target="_blank">Best pose PDBQT</a><a href="/artifact/${d.result.complex_pdb}" target="_blank">Complex PDB</a>`;if(d.result.complex_png){$('dockimg').src='/artifact/'+d.result.complex_png+'?t='+Date.now();$('dockimg').classList.remove('hidden')}$('dockstatus').textContent='Best pocket: '+d.result.best_binding_site+' | '+Number(d.result.best_affinity_kcal_mol).toFixed(3)+' kcal/mol'}catch(e){$('dockstatus').textContent=e.message}finally{$('dock').disabled=false}}
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, payload: bytes, content_type: str, status=200):
        self.send_response(status); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/health":
            self.send_bytes(json.dumps({"status": "ok", "models": 3}).encode(), "application/json")
        elif path.startswith("/api/dock/"):
            job_id = path.rsplit("/", 1)[-1]
            with DOCK_LOCK: job = DOCK_JOBS.get(job_id)
            if job is None:
                self.send_bytes(json.dumps({"error": "Unknown job"}).encode(), "application/json", 404)
            else:
                self.send_bytes(json.dumps(job).encode(), "application/json")
        elif path.startswith("/artifact/"):
            rel = path[len("/artifact/"):]
            candidate = (ROOT / rel).resolve()
            allowed = (ROOT / "results" / "web_docking").resolve()
            if allowed not in candidate.parents or not candidate.is_file():
                self.send_bytes(b"Not found", "text/plain", 404); return
            ctype = "image/png" if candidate.suffix == ".png" else "application/octet-stream"
            self.send_bytes(candidate.read_bytes(), ctype)
        else: self.send_bytes(b"Not found", "text/plain", 404)

    def do_POST(self):
        if self.path not in ("/api/predict", "/api/dock"):
            return self.send_bytes(b"Not found", "text/plain", 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            smiles = str(body.get("smiles", ""))
            if self.path == "/api/predict":
                self.send_bytes(json.dumps(predict(smiles)).encode(), "application/json")
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None: raise ValueError("Invalid SMILES")
                canonical = Chem.MolToSmiles(mol, canonical=True)
                job_id = "WEB_" + hashlib.sha256(canonical.encode()).hexdigest()[:12].upper()
                with DOCK_LOCK:
                    existing = DOCK_JOBS.get(job_id)
                    if not existing or existing.get("status") == "failed":
                        DOCK_JOBS[job_id] = {"job_id": job_id, "status": "queued", "message": "Queued"}
                        def worker():
                            def update(message):
                                with DOCK_LOCK:
                                    DOCK_JOBS[job_id].update(status="running", message=message)
                            try:
                                result = WEB_DOCK.dock_smiles(ROOT, canonical, job_id, update)
                                with DOCK_LOCK:
                                    DOCK_JOBS[job_id].update(status="complete", message="Docking complete", result=result)
                            except Exception as exc:
                                with DOCK_LOCK:
                                    DOCK_JOBS[job_id].update(status="failed", message="Docking failed", error=str(exc))
                        threading.Thread(target=worker, daemon=True).start()
                self.send_bytes(json.dumps({"job_id": job_id}).encode(), "application/json", 202)
        except Exception as exc:
            self.send_bytes(json.dumps({"error": str(exc)}).encode(), "application/json", 400)

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} {fmt % args}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--open", action="store_true")
    args = ap.parse_args(); url = f"http://{args.host}:{args.port}"
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"CFTR Molecule Designer is already running at {url}")
            if args.open: webbrowser.open(url)
            return
        raise
    print(f"CFTR Molecule Designer running at {url}")
    if args.open: threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
