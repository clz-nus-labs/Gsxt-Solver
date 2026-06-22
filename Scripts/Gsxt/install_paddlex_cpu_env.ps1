$ErrorActionPreference = "Stop"

$envName = "paddlex_cv"

Write-Host "Creating conda environment: $envName"
conda create -n $envName python=3.10 -y

Write-Host "Installing PaddleX CPU dependencies into $envName"
conda run -n $envName python -m pip install --upgrade pip
conda run -n $envName python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
conda run -n $envName python -m pip install "paddlex[cv]"
conda run -n $envName python -m pip install paddleocr
conda run -n $envName python -m pip install opencv-python numpy Pillow
conda run -n $envName python -m pip install tqdm

Write-Host "Checking Paddle installation"
conda run -n $envName python -c "import paddle; print('Paddle version:', paddle.__version__); paddle.utils.run_check()"

Write-Host "Done. Activate it with:"
Write-Host "conda activate $envName"
