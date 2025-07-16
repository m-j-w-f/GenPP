import xarray as xr


def print_info(data: xr.Dataset) -> None:
    print(f"Subset shape: {data.sizes}")
    size_gb = data.nbytes / (1024**3)  # Convert bytes to gigabytes
    print(f"Subset size: {size_gb:.2f} GB")
