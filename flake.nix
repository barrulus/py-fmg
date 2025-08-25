{
  description = "Python Fantasy Map Generator development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # Core dependencies for geospatial work
          numpy
          scipy
          shapely
          geopandas
          rasterio
          psycopg2
          
          # Web framework
          fastapi
          uvicorn
          
          # Development tools
          pytest
          black
          isort
          mypy
          ruff
          bandit
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python environment
            pythonEnv
            poetry
            
            
            # Geospatial libraries
            gdal
            geos
            proj
            
            # Development tools
            git
          ];

          shellHook = ''
            echo "Python Fantasy Map Generator development environment"
            echo "Python: $(python --version)"
            echo "Poetry: $(poetry --version)"
            
            # Set up environment variables for geospatial libraries
            export GDAL_DATA="${pkgs.gdal}/share/gdal"
            export PROJ_LIB="${pkgs.proj}/share/proj"
          '';
        };
      });
}