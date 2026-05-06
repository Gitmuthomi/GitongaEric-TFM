from config import get_config


def main() -> None:
    """
    Print resolved configuration and validate that critical paths exist.
    """
    cfg = get_config()

    print("=" * 60)
    print("Resolved project configuration")
    print("=" * 60)

    path_keys = ["project_dir", "data_dir", "output_dir", "ssl4eo_weights"]
    scalar_keys = ["conda_env", "seeds", "num_classes", "ignore_label",
                   "patch_size", "num_bands"]

    print("\nPaths:")
    for key in path_keys:
        path = cfg[key]
        status = "OK" if path.exists() else "MISSING"
        print(f"  {key:<20} {status}  {path}")

    print("\nConstants:")
    for key in scalar_keys:
        print(f"  {key:<20} {cfg[key]}")

    print()

    missing = [k for k in path_keys if not cfg[k].exists()]
    # output_dir missing since it will be created by the training scripts.
    missing = [k for k in missing if k != "output_dir"]

    if missing:
        print(f"WARNING: {len(missing)} path(s) not found on disk:")
        for k in missing:
            print(f"  {k}: {cfg[k]}")
        print("\nCheck your .env file (copy from .env.example if absent).")
    else:
        print("All critical paths resolved. Configuration looks good.")


if __name__ == "__main__":
    main()
