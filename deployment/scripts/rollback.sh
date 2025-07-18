#!/bin/bash

# Rollback Script for Academic Agent
# Provides automated rollback capabilities with backup restoration

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="academic-agent"
BACKUP_ROOT="/backup"

# Default values
ROLLBACK_TYPE="${1:-deployment}"  # deployment, database, full
BACKUP_ID="${2:-}"                # Specific backup to restore from
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Function to list available backups
list_backups() {
    log "Available backups:"
    
    # List local backups
    echo ""
    info "Local backups:"
    if ls "$BACKUP_ROOT"/academic-agent-*.tar.gz 2>/dev/null; then
        for backup in "$BACKUP_ROOT"/academic-agent-*.tar.gz; do
            if [[ -f "$backup" ]]; then
                backup_name=$(basename "$backup" .tar.gz)
                backup_date=$(echo "$backup_name" | sed 's/academic-agent-\([0-9_]*\)/\1/' | sed 's/_/ /')
                backup_size=$(du -sh "$backup" | cut -f1)
                echo "  - $backup_name (Date: $backup_date, Size: $backup_size)"
            fi
        done
    else
        echo "  No local backups found"
    fi
    
    # List S3 backups if configured
    if [[ -n "${S3_BUCKET:-}" ]]; then
        echo ""
        info "S3 backups:"
        if aws s3 ls "s3://$S3_BUCKET/${S3_PREFIX:-backups/academic-agent}/" 2>/dev/null | grep "academic-agent-"; then
            aws s3 ls "s3://$S3_BUCKET/${S3_PREFIX:-backups/academic-agent}/" | \
                grep "academic-agent-" | \
                awk '{print "  - " $4 " (Date: " $1 " " $2 ", Size: " $3 ")"}'
        else
            echo "  No S3 backups found"
        fi
    fi
}

# Function to find the latest backup
find_latest_backup() {
    local latest_backup=""
    
    # Find latest local backup
    for backup in "$BACKUP_ROOT"/academic-agent-*.tar.gz; do
        if [[ -f "$backup" ]]; then
            if [[ -z "$latest_backup" || "$backup" -nt "$latest_backup" ]]; then
                latest_backup="$backup"
            fi
        fi
    done
    
    if [[ -n "$latest_backup" ]]; then
        basename "$latest_backup" .tar.gz
    fi
}

# Function to download backup from S3
download_backup_from_s3() {
    local backup_id="$1"
    
    if [[ -z "${S3_BUCKET:-}" ]]; then
        error "S3_BUCKET not configured"
        return 1
    fi
    
    log "Downloading backup from S3: $backup_id"
    
    # Download backup file
    aws s3 cp "s3://$S3_BUCKET/${S3_PREFIX:-backups/academic-agent}/$backup_id.tar.gz" \
        "$BACKUP_ROOT/$backup_id.tar.gz"
    
    # Download and verify checksums
    aws s3 cp "s3://$S3_BUCKET/${S3_PREFIX:-backups/academic-agent}/$backup_id.tar.gz.sha256" \
        "$BACKUP_ROOT/$backup_id.tar.gz.sha256"
    
    # Verify checksum
    cd "$BACKUP_ROOT"
    if ! sha256sum -c "$backup_id.tar.gz.sha256"; then
        error "Checksum verification failed for $backup_id"
        return 1
    fi
    
    log "Backup downloaded and verified: $backup_id"
}

# Function to extract backup
extract_backup() {
    local backup_id="$1"
    local backup_file="$BACKUP_ROOT/$backup_id.tar.gz"
    
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file not found: $backup_file"
        return 1
    fi
    
    log "Extracting backup: $backup_id"
    
    cd "$BACKUP_ROOT"
    tar -xzf "$backup_id.tar.gz"
    
    if [[ ! -d "$BACKUP_ROOT/$backup_id" ]]; then
        error "Backup extraction failed"
        return 1
    fi
    
    log "Backup extracted to: $BACKUP_ROOT/$backup_id"
}

# Function to validate backup
validate_backup() {
    local backup_dir="$1"
    
    log "Validating backup: $backup_dir"
    
    # Check if backup manifest exists
    if [[ ! -f "$backup_dir/backup_manifest.json" ]]; then
        error "Backup manifest not found"
        return 1
    fi
    
    # Parse manifest
    local has_database=$(jq -r '.components.database' "$backup_dir/backup_manifest.json")
    local has_redis=$(jq -r '.components.redis' "$backup_dir/backup_manifest.json")
    local has_config=$(jq -r '.components.config' "$backup_dir/backup_manifest.json")
    
    info "Backup validation:"
    echo "  - Database: $has_database"
    echo "  - Redis: $has_redis"
    echo "  - Configuration: $has_config"
    
    # Verify critical files exist
    if [[ "$has_database" == "true" ]]; then
        if [[ ! -f "$backup_dir/database/academic_agent_prod.dump" ]]; then
            error "Database dump file missing"
            return 1
        fi
    fi
    
    if [[ "$has_redis" == "true" ]]; then
        if [[ ! -f "$backup_dir/redis/dump.rdb" ]]; then
            error "Redis dump file missing"
            return 1
        fi
    fi
    
    log "Backup validation passed"
}

# Function to rollback deployment
rollback_deployment() {
    log "Rolling back deployment..."
    
    # Check if deployment exists
    if ! kubectl get deployment academic-agent-app -n "$NAMESPACE" &> /dev/null; then
        error "Application deployment not found"
        return 1
    fi
    
    # Get current deployment info
    info "Current deployment status:"
    kubectl rollout history deployment/academic-agent-app -n "$NAMESPACE"
    
    # Confirm rollback
    if [[ "$FORCE" != "true" && "$DRY_RUN" != "true" ]]; then
        echo ""
        read -p "Are you sure you want to rollback the deployment? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            warn "Rollback cancelled by user"
            return 0
        fi
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would rollback deployment"
        return 0
    fi
    
    # Perform rollback
    info "Performing deployment rollback..."
    kubectl rollout undo deployment/academic-agent-app -n "$NAMESPACE"
    
    # Wait for rollback to complete
    info "Waiting for rollback to complete..."
    kubectl rollout status deployment/academic-agent-app -n "$NAMESPACE" --timeout=300s
    
    # Verify rollback
    info "Verifying rollback..."
    kubectl get pods -n "$NAMESPACE" -l app=academic-agent
    
    log "Deployment rollback completed"
}

# Function to restore database
restore_database() {
    local backup_dir="$1"
    
    log "Restoring database from backup..."
    
    # Check if database backup exists
    if [[ ! -f "$backup_dir/database/academic_agent_prod.dump" ]]; then
        error "Database backup not found"
        return 1
    fi
    
    # Check if PostgreSQL deployment exists
    if ! kubectl get deployment academic-agent-postgres -n "$NAMESPACE" &> /dev/null; then
        error "PostgreSQL deployment not found"
        return 1
    fi
    
    # Get PostgreSQL pod
    POSTGRES_POD=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$POSTGRES_POD" ]]; then
        error "No PostgreSQL pod found"
        return 1
    fi
    
    # Confirm database restore
    if [[ "$FORCE" != "true" && "$DRY_RUN" != "true" ]]; then
        echo ""
        warn "This will COMPLETELY REPLACE the current database!"
        read -p "Are you sure you want to restore the database? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            warn "Database restore cancelled by user"
            return 0
        fi
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would restore database from $backup_dir/database/academic_agent_prod.dump"
        return 0
    fi
    
    # Stop application pods to prevent database access during restore
    info "Scaling down application pods..."
    kubectl scale deployment academic-agent-app -n "$NAMESPACE" --replicas=0
    
    # Wait for pods to terminate
    kubectl wait --for=delete pod -l app=academic-agent -n "$NAMESPACE" --timeout=60s || true
    
    # Create current database backup before restore
    info "Creating safety backup of current database..."
    kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- pg_dump \
        -U academic_agent \
        -d academic_agent_prod \
        --no-password \
        --format=custom > "/tmp/pre-restore-backup-$(date +%Y%m%d_%H%M%S).dump"
    
    # Restore database
    info "Restoring database..."
    kubectl exec -i -n "$NAMESPACE" "$POSTGRES_POD" -- pg_restore \
        -U academic_agent \
        -d academic_agent_prod \
        --clean \
        --if-exists \
        --no-password \
        --verbose < "$backup_dir/database/academic_agent_prod.dump"
    
    # Scale application pods back up
    info "Scaling up application pods..."
    kubectl scale deployment academic-agent-app -n "$NAMESPACE" --replicas=2
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=academic-agent -n "$NAMESPACE" --timeout=300s
    
    log "Database restore completed"
}

# Function to restore Redis
restore_redis() {
    local backup_dir="$1"
    
    log "Restoring Redis from backup..."
    
    # Check if Redis backup exists
    if [[ ! -f "$backup_dir/redis/dump.rdb" ]]; then
        error "Redis backup not found"
        return 1
    fi
    
    # Check if Redis deployment exists
    if ! kubectl get deployment academic-agent-redis -n "$NAMESPACE" &> /dev/null; then
        error "Redis deployment not found"
        return 1
    fi
    
    # Get Redis pod
    REDIS_POD=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$REDIS_POD" ]]; then
        error "No Redis pod found"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would restore Redis from $backup_dir/redis/dump.rdb"
        return 0
    fi
    
    # Create current Redis backup
    info "Creating safety backup of current Redis data..."
    kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli BGSAVE
    kubectl cp "$NAMESPACE/$REDIS_POD:/data/dump.rdb" "/tmp/redis-pre-restore-backup-$(date +%Y%m%d_%H%M%S).rdb"
    
    # Stop Redis temporarily
    info "Stopping Redis temporarily..."
    kubectl scale deployment academic-agent-redis -n "$NAMESPACE" --replicas=0
    kubectl wait --for=delete pod -l app=redis -n "$NAMESPACE" --timeout=60s || true
    
    # Copy backup to Redis data volume
    info "Restoring Redis data..."
    kubectl run redis-restore-pod -n "$NAMESPACE" --image=busybox --rm -i --restart=Never \
        --overrides='{"spec":{"containers":[{"name":"redis-restore-pod","image":"busybox","command":["sh","-c","cp /backup/dump.rdb /data/ && chmod 644 /data/dump.rdb"],"volumeMounts":[{"name":"redis-data","mountPath":"/data"},{"name":"backup-data","mountPath":"/backup"}]}],"volumes":[{"name":"redis-data","persistentVolumeClaim":{"claimName":"redis-pvc"}},{"name":"backup-data","hostPath":{"path":"'$backup_dir/redis'"}}]}}' \
        --wait=true
    
    # Restart Redis
    info "Starting Redis..."
    kubectl scale deployment academic-agent-redis -n "$NAMESPACE" --replicas=1
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s
    
    log "Redis restore completed"
}

# Function to restore configuration
restore_configuration() {
    local backup_dir="$1"
    
    log "Restoring configuration from backup..."
    
    # Check if configuration backup exists
    if [[ ! -f "$backup_dir/config/all_resources.yaml" ]]; then
        error "Configuration backup not found"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would restore configuration from $backup_dir/config/"
        return 0
    fi
    
    # Backup current configuration
    info "Creating safety backup of current configuration..."
    kubectl get all -n "$NAMESPACE" -o yaml > "/tmp/current-config-backup-$(date +%Y%m%d_%H%M%S).yaml"
    
    # Apply configuration
    info "Applying configuration..."
    kubectl apply -f "$backup_dir/config/all_resources.yaml"
    
    # Wait for deployments to be ready
    info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available deployment --all -n "$NAMESPACE" --timeout=300s
    
    log "Configuration restore completed"
}

# Function to perform full rollback
full_rollback() {
    local backup_dir="$1"
    
    log "Performing full system rollback..."
    
    # Confirm full rollback
    if [[ "$FORCE" != "true" && "$DRY_RUN" != "true" ]]; then
        echo ""
        error "WARNING: This will COMPLETELY REPLACE all system components!"
        warn "This includes database, Redis, configuration, and application deployment."
        echo ""
        read -p "Are you absolutely sure you want to perform a full rollback? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            warn "Full rollback cancelled by user"
            return 0
        fi
    fi
    
    # Perform rollback in order
    restore_configuration "$backup_dir"
    restore_database "$backup_dir"
    restore_redis "$backup_dir"
    
    # Final health check
    info "Running post-rollback health checks..."
    sleep 30
    
    # Check if all pods are ready
    kubectl wait --for=condition=ready pod --all -n "$NAMESPACE" --timeout=300s
    
    log "Full rollback completed"
}

# Function to show rollback status
show_status() {
    log "System Status After Rollback:"
    
    echo ""
    info "Namespace: $NAMESPACE"
    kubectl get all -n "$NAMESPACE"
    
    echo ""
    info "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    info "Recent Events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# Main rollback function
main() {
    log "Starting Academic Agent rollback - Type: $ROLLBACK_TYPE"
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Handle list command
    if [[ "$ROLLBACK_TYPE" == "list" ]]; then
        list_backups
        exit 0
    fi
    
    # Handle deployment-only rollback
    if [[ "$ROLLBACK_TYPE" == "deployment" ]]; then
        rollback_deployment
        show_status
        exit 0
    fi
    
    # For database and full rollbacks, we need a backup
    local backup_id="$BACKUP_ID"
    local backup_dir=""
    
    # Find backup to use
    if [[ -z "$backup_id" ]]; then
        backup_id=$(find_latest_backup)
        if [[ -z "$backup_id" ]]; then
            error "No backups found and no backup ID specified"
            info "Use: $0 list to see available backups"
            exit 1
        fi
        warn "Using latest backup: $backup_id"
    fi
    
    # Check if backup exists locally
    if [[ ! -f "$BACKUP_ROOT/$backup_id.tar.gz" ]]; then
        info "Backup not found locally, attempting to download from S3..."
        download_backup_from_s3 "$backup_id"
    fi
    
    # Extract and validate backup
    extract_backup "$backup_id"
    backup_dir="$BACKUP_ROOT/$backup_id"
    validate_backup "$backup_dir"
    
    # Perform the requested rollback
    case "$ROLLBACK_TYPE" in
        "database")
            restore_database "$backup_dir"
            ;;
        "redis")
            restore_redis "$backup_dir"
            ;;
        "config")
            restore_configuration "$backup_dir"
            ;;
        "full")
            full_rollback "$backup_dir"
            ;;
        *)
            error "Invalid rollback type: $ROLLBACK_TYPE"
            usage
            exit 1
            ;;
    esac
    
    # Show final status
    show_status
    
    log "Rollback completed successfully"
}

# Script usage
usage() {
    echo "Usage: $0 [type] [backup_id]"
    echo ""
    echo "Types:"
    echo "  deployment  - Rollback only the application deployment (default)"
    echo "  database    - Restore database from backup"
    echo "  redis       - Restore Redis from backup"
    echo "  config      - Restore configuration from backup"
    echo "  full        - Full system rollback (database + redis + config)"
    echo "  list        - List available backups"
    echo ""
    echo "Options:"
    echo "  backup_id   - Specific backup to restore (optional, uses latest if not specified)"
    echo ""
    echo "Environment variables:"
    echo "  DRY_RUN     - Set to 'true' for dry run (default: false)"
    echo "  FORCE       - Set to 'true' to skip confirmations (default: false)"
    echo "  S3_BUCKET   - S3 bucket for backup storage (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 deployment"
    echo "  $0 database academic-agent-20231201_120000"
    echo "  $0 full"
    echo "  DRY_RUN=true $0 full academic-agent-20231201_120000"
}

# Parse command line arguments
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

# Validate rollback type
case "${1:-deployment}" in
    "deployment"|"database"|"redis"|"config"|"full"|"list")
        ;;
    *)
        error "Invalid rollback type: ${1:-}"
        usage
        exit 1
        ;;
esac

# Run main function
main "$@"