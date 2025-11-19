#!/bin/bash

# Script to drop SQLite database
# Usage: ./drop_db.sh

set -e

# Configuration - UPDATE THIS
DB_FILE="${DB_FILE:-./database.db}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Drop SQLite Database ===${NC}\n"

if [ ! -f "$DB_FILE" ]; then
    echo -e "${RED}Database file not found: $DB_FILE${NC}"
    exit 1
fi

echo -e "${RED}WARNING: This will permanently delete: $DB_FILE${NC}"
read -p "Are you sure? (yes/no): " confirmation

if [ "$confirmation" != "yes" ]; then
    echo -e "${YELLOW}Aborted.${NC}"
    exit 0
fi

rm -f "$DB_FILE"
echo -e "${GREEN}✓ Database deleted: $DB_FILE${NC}"

# Also remove WAL and SHM files if they exist
rm -f "${DB_FILE}-wal" "${DB_FILE}-shm"
echo -e "${GREEN}✓ Cleanup complete${NC}"
