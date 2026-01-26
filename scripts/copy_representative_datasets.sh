#!/bin/bash
# Copy representative datasets from st_semantics to semanscope

SRC_DIR="/home/papagame/projects/Proj-Geometry-of-Meaning/st_semantics/data/input"
DEST_DIR="/home/papagame/projects/Proj-Geometry-of-Meaning/semanscope/data/input"

echo "Copying representative datasets..."

# ACL-4 Numbers
cp "$SRC_DIR/ACL-4-Numbers-enu.txt" "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-4-Numbers-enu.txt"
cp "$SRC_DIR/ACL-4-Numbers-chn.txt" "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-4-Numbers-chn.txt"
cp "$SRC_DIR/ACL-4-emotions-enu.txt" "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-4-emotions-enu.txt"
cp "$SRC_DIR/ACL-4-animals-enu.txt" "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-4-animals-enu.txt"

# ACL-5 Poems (representative samples)
cp "$SRC_DIR/ACL-5-Poems-LiBai-chn.txt" "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-5-Poems-LiBai-chn.txt"
cp "$SRC_DIR/ACL-5-Poems-DuFu-chn.txt" "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-5-Poems-DuFu-chn.txt"
cp "$SRC_DIR/ACL-5-Poems-Frost-enu.txt" "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-5-Poems-Frost-enu.txt"

# ACL-6 Emoji and Pictographs
cp "$SRC_DIR/ACL-6-Emoji-"* "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-6-Emoji files"
cp "$SRC_DIR/ACL-6-Pictographs-"* "$DEST_DIR/" 2>/dev/null || echo "Skipped: ACL-6-Pictographs files"

# NeurIPS datasets (v2.5 - latest versions)
echo "Copying NeurIPS benchmark datasets (v2.5)..."
for i in {01..11}; do
    cp "$SRC_DIR/NeurIPS-$i-"*"-v2.5-"*".txt" "$DEST_DIR/" 2>/dev/null || echo "Skipped: NeurIPS-$i"
done

# Copy color code files
echo "Copying color code files..."
cp "$SRC_DIR"/*.color-code.csv "$DEST_DIR/" 2>/dev/null || echo "No color-code files found"

echo "Dataset copy complete!"
echo "Listing copied files:"
ls -1 "$DEST_DIR" | wc -l
echo "files copied to $DEST_DIR"
