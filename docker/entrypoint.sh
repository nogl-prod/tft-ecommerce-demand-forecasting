#!/bin/bash
# Entrypoint script for downloading code from Garage at runtime
# This script handles code download, extraction, and execution

set -e  # Exit on error

# Configuration from environment variables
GARAGE_ENDPOINT="${GARAGE_ENDPOINT:-http://192.168.29.163:3900}"
GARAGE_BUCKET="${GARAGE_CODE_BUCKET:-code-repository}"
CODE_VERSION="${CODE_VERSION:-latest}"
CODE_ARCHIVE="/tmp/code-${CODE_VERSION}.tar.gz"
CODE_DIR="/workspace/code"
MAX_RETRIES=3
RETRY_DELAY=5

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Download code from Garage with retry logic
download_code() {
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        log "Downloading code from Garage (attempt $((retry_count + 1))/$MAX_RETRIES)..."
        log "Endpoint: $GARAGE_ENDPOINT"
        log "Bucket: $GARAGE_BUCKET"
        log "Version: $CODE_VERSION"
        
        if aws s3 cp \
            "s3://${GARAGE_BUCKET}/code-${CODE_VERSION}.tar.gz" \
            "$CODE_ARCHIVE" \
            --endpoint-url "$GARAGE_ENDPOINT" \
            --region garage \
            --quiet; then
            log "Code download successful"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log "Download failed, retrying in ${RETRY_DELAY} seconds..."
                sleep $RETRY_DELAY
            else
                log "ERROR: Failed to download code after $MAX_RETRIES attempts"
                return 1
            fi
        fi
    done
}

# Extract code archive
extract_code() {
    log "Extracting code archive..."
    
    if [ ! -f "$CODE_ARCHIVE" ]; then
        log "ERROR: Code archive not found: $CODE_ARCHIVE"
        return 1
    fi
    
    # Remove existing code directory if it exists
    if [ -d "$CODE_DIR" ] && [ "$(ls -A $CODE_DIR)" ]; then
        log "Cleaning existing code directory..."
        rm -rf "$CODE_DIR"/*
    fi
    
    # Extract archive
    tar -xzf "$CODE_ARCHIVE" -C "$CODE_DIR" || {
        log "ERROR: Failed to extract code archive"
        return 1
    }
    
    log "Code extraction successful"
    
    # Clean up archive
    rm -f "$CODE_ARCHIVE"
    
    return 0
}

# Verify code was extracted
verify_code() {
    log "Verifying code extraction..."
    
    if [ ! -d "$CODE_DIR" ] || [ -z "$(ls -A $CODE_DIR)" ]; then
        log "ERROR: Code directory is empty or does not exist"
        return 1
    fi
    
    log "Code verification successful"
    log "Code directory contents:"
    ls -la "$CODE_DIR" | head -10
    
    return 0
}

# Main execution
main() {
    log "=== Code Download Entrypoint ==="
    log "Starting code download process..."
    
    # Check if code already exists (caching)
    if [ -d "$CODE_DIR" ] && [ -n "$(ls -A $CODE_DIR)" ] && [ "$CODE_VERSION" = "latest" ]; then
        log "Code directory already exists, skipping download (cache hit)"
    else
        # Download and extract code
        download_code || exit 1
        extract_code || exit 1
        verify_code || exit 1
    fi
    
    log "=== Code Ready ==="
    log "Executing command: $*"
    
    # Change to code directory and execute command
    cd "$CODE_DIR"
    exec "$@"
}

# Run main function with all arguments
main "$@"

