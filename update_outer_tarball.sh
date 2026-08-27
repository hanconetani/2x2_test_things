#!/bin/bash
# update_outer_tarball.sh
# Rebuilds mypipeline.tar.gz after editing scripts in Python_Scripts/.
# Does NOT touch myenv.tar.gz -- only run this for script edits, not package changes.

set -e   # stop immediately if anything fails, rather than continuing on a bad state

PIPELINE_DIR="/exp/dune/app/users/hanconet/LRS_VBR_pipeline"
cd "$PIPELINE_DIR"

echo "Rebuilding mypipeline.tar.gz from Python_Scripts/ and myenv.tar.gz ..."
tar --exclude='.git' --exclude='*.pyc' --exclude='__pycache__' \
    -czf mypipeline.tar.gz Python_Scripts/ myenv.tar.gz

echo ""
echo "Done. Contents of the new tarball:"
tar -tzf mypipeline.tar.gz

echo ""
echo "Size check:"
ls -lh mypipeline.tar.gz myenv.tar.gz