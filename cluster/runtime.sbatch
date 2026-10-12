#!/usr/bin/env bash
#SBATCH --job-name=loopwatch-runtime
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runs/runtime-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load miniforge/23.11.0-0
export CONDA_PKGS_DIRS="$PWD/.cache/conda-pkgs"
if [[ ! -x .cache/runtime/bin/python ]]; then
    mamba create -y -p "$PWD/.cache/runtime" -c conda-forge \
        python=3.11 pip pyyaml python-dotenv
fi
mkdir -p .cache/qemu-rpms .cache/qemu-root-minimal
dnf -y --setopt=cachedir="$PWD/.cache/dnf-cache" \
    --setopt=persistdir="$PWD/.cache/dnf-state" \
    download --resolve --destdir="$PWD/.cache/qemu-rpms" qemu-kvm-core qemu-img
for package in "$PWD"/.cache/qemu-rpms/*.rpm; do
    (cd .cache/qemu-root-minimal && rpm2cpio "$package" | cpio -idm --quiet --no-preserve-owner)
done
cat >.cache/runtime/bin/qemu-system-x86_64 <<'SH'
#!/usr/bin/env bash
set -euo pipefail
runtime_root="$(cd "$(dirname "$0")/../../qemu-root-minimal" && pwd)"
export LD_LIBRARY_PATH="$runtime_root/usr/lib64"
exec "$runtime_root/usr/libexec/qemu-kvm" -L "$runtime_root/usr/share/qemu-kvm" "$@"
SH
cat >.cache/runtime/bin/qemu-img <<'SH'
#!/usr/bin/env bash
set -euo pipefail
runtime_root="$(cd "$(dirname "$0")/../../qemu-root-minimal" && pwd)"
export LD_LIBRARY_PATH="$runtime_root/usr/lib64"
exec "$runtime_root/usr/bin/qemu-img" "$@"
SH
chmod +x .cache/runtime/bin/qemu-system-x86_64 .cache/runtime/bin/qemu-img
.cache/runtime/bin/qemu-system-x86_64 --version
.cache/runtime/bin/qemu-img --version
.cache/runtime/bin/python -m pip install -e .
