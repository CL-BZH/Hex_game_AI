#!/bin/bash
echo "Running clang-format..."
find . -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | xargs clang-format -i --style=file
#git add -u
