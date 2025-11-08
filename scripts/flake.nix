{
  description = "Python dev shell with matplotlib";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    pkgs = import nixpkgs {system = "x86_64-linux";};
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {
      buildInputs = [
        pkgs.python313
        pkgs.python313Packages.matplotlib
        pkgs.python313Packages.pandas
        pkgs.python313Packages.seaborn
        pkgs.python313Packages.scipy
        pkgs.python313Packages.pip
      ];
    };
  };
}
