# Run the spiral benchmark on a TPU Builder VM

Use a running, available TPU in the credited project first. The benchmark uses one process on one worker and addresses the default TPU device. The Builder guide's v6e example allocates four chips on one VM. All four chips count toward the bill. At the listed us-east5 Flex-start rate on November 18, 2026, one hour costs about **$5.40 in TPU charges** (4 × $1.35), plus applicable disk and network charges. Check the credit's remaining value, expiry and eligible services in the project's linked billing account. [Builder guide](https://docs.google.com/document/d/1u2W79sxupb1Nm53zONbsnjyHzwcjuu-76urThl-MhRs/edit) · [v6e configurations](https://docs.cloud.google.com/tpu/docs/v6e) · [TPU pricing](https://cloud.google.com/tpu/pricing) · [Billing credits](https://docs.cloud.google.com/billing/docs/how-to/resolve-issues)

## Choose the allocation in Cloud Shell

Open Cloud Shell from the credited project. Run these Bash commands, replacing the placeholder. Check Compute Engine for an available v6e/v5p instance. For a known legacy v5e allocation, list TPU nodes in its zone separately. Accessing a teammate's VM requires an available worker and their agreement.

~~~bash
export PROJECT_ID=YOUR_CREDITED_PROJECT_ID
export ZONE=us-east5-a
gcloud auth list --filter=status:ACTIVE --format='value(account)'
gcloud billing projects describe "$PROJECT_ID"
gcloud services list --enabled --project="$PROJECT_ID" --format='value(config.name)'
gcloud compute instances list --project="$PROJECT_ID" \
  --filter='machineType ~ "/ct(5p|6e)-"' --format='table(name,zone.basename(),machineType.basename(),status)'
# Legacy v5e in the Builder program, if applicable
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone=us-west4-a
~~~

For an existing Compute Engine TPU, export its actual VM name and zone, set **TPU_API=compute**, and skip to **Install TorchTPU on the worker**. For an existing legacy TPU, export the *node* name and its zone, then set **TPU_API=legacy**. Worker 0 is the runner's sole target. The July 2026 Builder guide lists us-west4-a and europe-west4-b for legacy v5e and uses a separate queued-resource API to request new v5e capacity. A v5p Compute Engine allocation also uses the compute path. [Compute Engine TPU setup](https://docs.cloud.google.com/tpu/docs/create-instance-compute) · [Legacy Flex-start requests](https://docs.cloud.google.com/tpu/docs/request-using-flex-start) · [Legacy worker transfers](https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/tpu-vm/scp)

~~~bash
export VM=EXISTING_VM_OR_NODE_NAME
export ZONE=ITS_ACTUAL_ZONE
export TPU_API=compute
~~~

## Request one v6e VM if needed

The guide lists us-east5-a, us-east5-b, us-central1-a, europe-west4-a and southamerica-west1-a for Builder v6e. Use a zone where your project has regional quota for four Flex-start v6e chips. Check **IAM & Admin → Quotas & System Limits** for both regional TPU and global allocation limits. Existing credits, quota and available capacity are separate. Enable the guide's APIs if missing. The data below are synthetic, and the job needs only the auto-deleting boot disk. [Compute Engine TPU setup](https://docs.cloud.google.com/tpu/docs/create-instance-compute) · [TPU storage](https://docs.cloud.google.com/tpu/docs/attach-durable-block-storage-compute)

~~~bash
export TPU_API=compute ZONE=us-east5-a
gcloud services enable compute.googleapis.com tpu.googleapis.com --project="$PROJECT_ID"
FAMILY=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e
gcloud compute images describe-from-family "$FAMILY" --project=ubuntu-os-accelerator-images --format='value(name)'
gcloud compute machine-types describe ct6e-standard-4t --zone="$ZONE" --project="$PROJECT_ID" --format='value(name)'
~~~

After both lookups succeed, request a single VM. This example shortens the guide's wait to 10 minutes and runtime to one hour. Compute Engine deletes the VM when the runtime expires. A stockout can still occur inside a valid quota. A pending request can be inspected by its exact name from a second Cloud Shell tab. [Flex-start lifetime and wait times](https://docs.cloud.google.com/compute/docs/instances/about-flex-start-vms)

~~~bash
export VM="lejepa-$(date -u +%Y%m%d-%H%M%S)-$RANDOM"
unset CREATED_ID
if gcloud compute instances create "$VM" --project="$PROJECT_ID" --zone="$ZONE" \
  --machine-type=ct6e-standard-4t --provisioning-model=FLEX_START \
  --request-valid-for-duration=10m --max-run-duration=1h --instance-termination-action=DELETE \
  --image-project=ubuntu-os-accelerator-images --image-family="$FAMILY" \
  --boot-disk-size=50GB --boot-disk-type=hyperdisk-balanced --boot-disk-auto-delete \
  --maintenance-policy=TERMINATE --labels=workload=lejepa-sprint; then
  CREATED_ID=$(gcloud compute instances describe "$VM" --project="$PROJECT_ID" --zone="$ZONE" --format='value(id)')
  gcloud compute instances describe "$VM" --project="$PROJECT_ID" --zone="$ZONE" \
    --format='yaml(name,id,status,machineType,scheduling,disks.autoDelete)'
fi
~~~

## Install TorchTPU on the worker

In Cloud Shell, open the corresponding SSH session. The commands assume normal SSH reachability for the project.

~~~bash
if [[ "$TPU_API" == legacy ]]; then
  gcloud compute tpus tpu-vm ssh "$VM" --worker=0 --project="$PROJECT_ID" --zone="$ZONE"
else
  gcloud compute ssh "$VM" --project="$PROJECT_ID" --zone="$ZONE"
fi
~~~

Inside the TPU VM, create a user-owned Python 3.12 environment at the path below. On the guide's Ubuntu 22.04 image, this bootstrap uses uv to obtain Python 3.12 if the system interpreter has a different version. Reuse an existing dedicated environment at that path when already configured. The TorchTPU wheel resolves its matching dependencies. The check executes and copies back a real TPU tensor. [TorchTPU VM installation](https://torchtpu.readthedocs.io/en/latest/get_started_on_a_vm/) · [Installing Python with uv](https://docs.astral.sh/uv/guides/install-python/)

~~~bash
TPU_ENV="$HOME/.venvs/lejepa-torchtpu"
if [[ ! -x "$TPU_ENV/bin/python" ]]; then
  sudo apt-get update && sudo apt-get install -y python3-venv
  mkdir -p "$HOME/.venvs"
  BOOTSTRAP=$(mktemp -d)
  python3 -m venv "$BOOTSTRAP"
  "$BOOTSTRAP/bin/python" -m pip install uv
  "$BOOTSTRAP/bin/uv" venv --python 3.12 --seed "$TPU_ENV"
  "$TPU_ENV/bin/python" -m pip install torch_tpu
fi
"$TPU_ENV/bin/python" -c 'import torch, torch_tpu; x = torch.arange(4, device="tpu"); print(torch.__version__, torch.tpu.get_device_properties().name, x.cpu().tolist())'
exit
~~~

A successful check prints the TPU device and the values 0, 1, 2, 3. If installation was interrupted, retry the pip install line inside the dedicated environment. After exiting SSH, the remaining steps run in Cloud Shell.

## Run and collect the evidence

Upload **run_experiment.py**, **spiral.py**, **run_cloud_suite.sh** and **report_results.py** to one directory in Cloud Shell and change into it. The first two files are the only source files sent to the TPU. The suite creates a unique run directory, logs each process, and copies its JSON back after each run. If the worker returns an error, inspect its log and JSON before spending more time. [Transfer files to Cloud Shell](https://docs.cloud.google.com/shell/docs/uploading-and-downloading-files)

~~~bash
bash run_cloud_suite.sh smoke 1337
~~~

After the two-step smoke run reports success, run a quick comparison of eager, fused eager and full-graph objective compilation. The benchmark uses 201 training steps per mode and starts a separate Python process for each, with persistent TorchTPU compilation caches disabled. The report compares modes when their recorded configuration and environment match and parity passes.

~~~bash
bash run_cloud_suite.sh benchmark 1337
~~~

For a longer crossover measurement, run the same seed on the same allocation. Each mode starts again from step 1, measures the first update separately, and times the remaining 1,000 updates in blocks of 20. This preserves first-use costs in each cumulative curve. The report identifies the measured interval where a mode pulls ahead of eager, or states that it never catches up through step 1,001. To repeat either preset with another seed, supply the same seed to both commands.

~~~bash
bash run_cloud_suite.sh long-benchmark 1337
~~~

For the scientific result, the **science** preset runs the matched eager full-objective and prediction-only control for one seed. It uses 6,000 steps, width 128, batch size 128, 64 projections and 50-step timing blocks. Use 1337 first. If the observed duration fits the remaining allocation time, use 1338 and 1339 for the other two CPU-reference seeds. Each seed is a separate invocation. A two-step result is a hardware check, and a 201-step run is a timing study.

~~~bash
bash run_cloud_suite.sh science 1337
~~~

The helper prints each downloaded directory under **cloud-runs/**. A status of **ok** in the JSON means loss and all parameter-gradient parity checks passed. Run the report on Cloud Shell or a workstation with matplotlib. The report keeps failed attempts visible and marks crossovers only at observed common step boundaries.

~~~bash
python3 -m venv .report-env
.report-env/bin/python -m pip install matplotlib
REPORT_DIR="cloud-report-$(date -u +%Y%m%d-%H%M%S)"
.report-env/bin/python report_results.py --runs cloud-runs --out "$REPORT_DIR"
ARCHIVE="$HOME/$REPORT_DIR.tar.gz"
tar -czf "$ARCHIVE" cloud-runs "$REPORT_DIR"
cloudshell download "$ARCHIVE"
~~~

## Release a VM created for this run

The local JSON and logs remain in Cloud Shell after the TPU is released. Check that the expected files have arrived. Release the newly created VM promptly using the same Cloud Shell session and its captured instance ID. Existing or borrowed allocations stay under their owner's lifecycle. The create command attaches one auto-delete boot disk. [Deleting a Compute Engine VM](https://docs.cloud.google.com/compute/docs/instances/deleting-instance)

~~~bash
if [[ -n "${CREATED_ID:-}" ]] && \
  [[ "$(gcloud compute instances describe "$VM" --project="$PROJECT_ID" --zone="$ZONE" --format='value(id)')" == "$CREATED_ID" ]]; then
  gcloud compute instances delete "$VM" --project="$PROJECT_ID" --zone="$ZONE"
fi
~~~

The command prompts for the named VM. If the runtime limit has already elapsed, the describe call reports that the resource is gone.
