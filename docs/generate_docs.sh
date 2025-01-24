# filepath: /home/gustavo/Documents/DataScience/llm/cvm-rag/docs/generate_docs.sh
#!/bin/bash

# Remove old generated files
rm -rf source/modules
mkdir -p source/modules

# Generate new .rst files
sphinx-apidoc -o source/modules ../cvm_rag

# Build the HTML documentation
make html