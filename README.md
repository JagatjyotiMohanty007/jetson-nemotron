# NVIDIA Nemotron on Jetson Orin Nano Super

Run NVIDIA Nemotron 3 Nano 4B locally on an 8 GB Jetson Orin Nano Super using CUDA-enabled llama.cpp and the official 2.84 GB Q4_K_M GGUF. The Python app uses the standard library.

## 1. Check the Jetson and NVMe

Extract or copy this folder to the NVMe and open a terminal in the project folder. Run these commands on the Jetson.

```bash
uname -m
head -n 1 /etc/nv_tegra_release
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS
findmnt -T "$PWD"
df -h .
```

Expect `aarch64` and Jetson Linux `R39`, revision `2.x` for JetPack 7.2.x. Confirm the project filesystem is on the NVMe. For a different JetPack release, use the NVIDIA instructions for the installed release before installing CUDA.

## 2. Install tools and build llama.cpp

```bash
sudo apt update
sudo apt install -y git curl build-essential cmake ninja-build python3
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
```

If `nvcc` is missing on JetPack 7.2.x, install its matching JetPack components and rerun `nvcc --version`.

```bash
sudo apt install -y nvidia-jetpack
```

Build for the Orin GPU's CUDA compute capability 8.7. Run these commands from the project folder.

```bash
git clone https://github.com/ggml-org/llama.cpp.git

cmake -S llama.cpp -B llama.cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build llama.cpp/build --parallel 2 --target llama-server
git -C llama.cpp rev-parse HEAD
```

Save the printed commit if you want to reproduce that exact build later.

## 3. Download and verify the model

```bash
mkdir -p models
MODEL="NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"
curl -fL --retry 3 -o "models/$MODEL" \
  "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/resolve/main/$MODEL"

printf '%s  %s\n' \
  'be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2' \
  "models/$MODEL" | sha256sum -c
```

Continue when the checksum shows `OK`.

## 4. Start inference and run the Python app

In the first terminal, from the project folder, run:

```bash
./llama.cpp/build/bin/llama-server \
  --model models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf \
  --alias nemotron-jetson \
  --n-gpu-layers 99 \
  --ctx-size 2048 \
  --parallel 1 \
  --reasoning off \
  --host 127.0.0.1 \
  --port 8080
```

Look for a CUDA device and GPU layer offload in the server log. If memory is exhausted, retry with `--ctx-size 1024` and `--n-gpu-layers 20`.

Leave the server running. Open a second terminal in the same project folder.

```bash
curl -fsS http://127.0.0.1:8080/health
python3 -m app
python3 -m app "Explain quantum superposition in two sentences."
```

A printed answer confirms the local Python request succeeded. To observe GPU activity while generating, run `tegrastats` in another terminal and watch `GR3D_FREQ`.

## 5. Push to GitHub

Create an empty repository named `jetson-nemotron` on GitHub. Leave the README, license, and gitignore options unchecked because this project already contains its own README and gitignore. Run these commands in the extracted project folder and replace `YOUR_USERNAME` with your account name.

```bash
git init -b main
git add README.md .gitignore app
git diff --cached --stat
git commit -m "Add Jetson Nemotron setup and Python demo"
git remote add origin https://github.com/JagatjyotiMohanty007/jetson-nemotron.git
git push -u origin main
```

The gitignore keeps the downloaded weights and llama.cpp checkout out of the commit.

## References

- [NVIDIA JetPack 7.2](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2)
- [Official NVIDIA model and checksum](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/blob/main/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf)
- [llama.cpp CUDA build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
