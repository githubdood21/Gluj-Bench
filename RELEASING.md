# Release process

1. Ensure the working tree contains only intended release changes.
2. Update the workspace version in `Cargo.toml` and the matching section in `CHANGELOG.md`.
3. Run the full local verification:

   ```powershell
   .\scripts\cargo.ps1 fmt --all -- --check
   .\scripts\cargo.ps1 clippy --workspace --all-targets '--' -D warnings
   .\scripts\cargo.ps1 test --workspace --locked
   .\scripts\package-release.ps1 -Version 0.3.1
   ```

4. Test the ZIP on a clean Windows x64 machine with current CPU and GPU drivers.
5. Commit and push the release changes. Wait for the `CI` workflow to pass.
6. Create and push an annotated semantic-version tag:

   ```powershell
   git tag -a v0.3.1 -m "Gluj-Bench 0.3.1"
   git push origin v0.3.1
   ```

The tag starts the `Release` workflow. It rebuilds and tests the workspace, creates a Windows x64 ZIP and SHA-256 checksum, and publishes a GitHub release with generated notes.

Confirm that the release archive contains `LICENSE` and that dependency licensing remains compatible with distribution.
