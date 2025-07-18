#!/bin/bash

# Backup Script for Academic Agent
# Performs automated backups of database, application data, and logs

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="academic-agent"
BACKUP_ROOT="/backup"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="$BACKUP_ROOT/academic-agent-$TIMESTAMP"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# S3 Configuration
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-backups/academic-agent}"

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

# Function to create backup directory
create_backup_dir() {
    log "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"/{database,redis,logs,config,metrics}
}

# Function to backup PostgreSQL database
backup_database() {
    log "Backing up PostgreSQL database..."
    
    # Check if PostgreSQL deployment exists
    if ! kubectl get deployment academic-agent-postgres -n "$NAMESPACE" &> /dev/null; then
        warn "PostgreSQL deployment not found, skipping database backup"
        return 0
    fi
    
    # Get PostgreSQL pod
    POSTGRES_POD=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$POSTGRES_POD" ]]; then
        error "No PostgreSQL pod found"
        return 1
    fi
    
    # Create database dump
    info "Creating database dump..."
    kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- pg_dump \
        -U academic_agent \
        -d academic_agent_prod \
        --no-password \
        --format=custom \
        --compress=9 \
        --verbose > "$BACKUP_DIR/database/academic_agent_prod.dump"
    
    # Create plain SQL dump for easier restoration
    kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- pg_dump \
        -U academic_agent \
        -d academic_agent_prod \
        --no-password \
        --format=plain \
        --verbose > "$BACKUP_DIR/database/academic_agent_prod.sql"
    
    # Backup database statistics
    kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql \
        -U academic_agent \
        -d academic_agent_prod \
        -c "SELECT schemaname, tablename, attname, n_distinct, correlation FROM pg_stats;" \
        --csv > "$BACKUP_DIR/database/statistics.csv"
    
    # Get database size information
    kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql \
        -U academic_agent \
        -d academic_agent_prod \
        -c "SELECT pg_size_pretty(pg_database_size('academic_agent_prod')) as database_size;" \
        --csv > "$BACKUP_DIR/database/size_info.csv"
    
    log "Database backup completed"
}

# Function to backup Redis data
backup_redis() {
    log "Backing up Redis data..."
    
    # Check if Redis deployment exists
    if ! kubectl get deployment academic-agent-redis -n "$NAMESPACE" &> /dev/null; then
        warn "Redis deployment not found, skipping Redis backup"
        return 0
    fi
    
    # Get Redis pod
    REDIS_POD=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$REDIS_POD" ]]; then
        error "No Redis pod found"
        return 1
    fi
    
    # Force Redis to save current state
    info "Forcing Redis save..."
    kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli BGSAVE
    
    # Wait for background save to complete
    while kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli LASTSAVE | xargs -I {} kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli LASTSAVE | grep -q {}; do
        sleep 1
    done
    
    # Copy Redis dump file
    info "Copying Redis dump file..."
    kubectl cp "$NAMESPACE/$REDIS_POD:/data/dump.rdb" "$BACKUP_DIR/redis/dump.rdb"
    
    # Copy AOF file if it exists
    if kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- ls /data/appendonly.aof &> /dev/null; then
        kubectl cp "$NAMESPACE/$REDIS_POD:/data/appendonly.aof" "$BACKUP_DIR/redis/appendonly.aof"
    fi
    
    # Get Redis configuration
    kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli CONFIG GET "*" > "$BACKUP_DIR/redis/config.txt"
    
    # Get Redis info
    kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli INFO > "$BACKUP_DIR/redis/info.txt"
    
    log "Redis backup completed"
}

# Function to backup application logs
backup_logs() {
    log "Backing up application logs..."
    
    # Check if application deployment exists
    if ! kubectl get deployment academic-agent-app -n "$NAMESPACE" &> /dev/null; then
        warn "Application deployment not found, skipping log backup"
        return 0
    fi
    
    # Get application pods
    APP_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=academic-agent -o jsonpath='{.items[*].metadata.name}')
    
    if [[ -z "$APP_PODS" ]]; then
        error "No application pods found"
        return 1
    fi
    
    # Backup logs from each pod
    for pod in $APP_PODS; do
        info "Backing up logs from pod: $pod"
        
        # Get current logs
        kubectl logs -n "$NAMESPACE" "$pod" --all-containers=true > "$BACKUP_DIR/logs/${pod}_current.log"
        
        # Get previous logs if available
        if kubectl logs -n "$NAMESPACE" "$pod" --previous --all-containers=true &> /dev/null; then
            kubectl logs -n "$NAMESPACE" "$pod" --previous --all-containers=true > "$BACKUP_DIR/logs/${pod}_previous.log"
        fi
    done
    
    # Backup persistent log volumes if they exist
    if kubectl get pvc app-logs-pvc -n "$NAMESPACE" &> /dev/null; then
        info "Copying persistent logs..."
        
        # Create a temporary pod to access the PVC
        kubectl run log-backup-pod -n "$NAMESPACE" --image=busybox --rm -i --restart=Never \
            --overrides='{"spec":{"containers":[{"name":"log-backup-pod","image":"busybox","command":["sh","-c","tar -czf - /logs"],"volumeMounts":[{"name":"logs","mountPath":"/logs"}]}],"volumes":[{"name":"logs","persistentVolumeClaim":{"claimName":"app-logs-pvc"}}]}}' \
            > "$BACKUP_DIR/logs/persistent_logs.tar.gz"
    fi
    
    log "Log backup completed"
}

# Function to backup configuration
backup_config() {
    log "Backing up configuration..."
    
    # Backup ConfigMaps
    info "Backing up ConfigMaps..."
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/config/configmaps.yaml"
    
    # Backup Secrets (without sensitive data)
    info "Backing up Secret metadata..."
    kubectl get secrets -n "$NAMESPACE" -o yaml | \
        sed 's/data:.*$/data: <REDACTED>/' > "$BACKUP_DIR/config/secrets_metadata.yaml"
    
    # Backup all resources
    info "Backing up all Kubernetes resources..."
    kubectl get all -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/config/all_resources.yaml"
    
    # Backup ingress
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        kubectl get ingress -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/config/ingress.yaml"
    fi
    
    # Backup PVCs
    if kubectl get pvc -n "$NAMESPACE" &> /dev/null; then
        kubectl get pvc -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/config/pvc.yaml"
    fi
    
    log "Configuration backup completed"
}

# Function to backup monitoring data
backup_monitoring() {
    log "Backing up monitoring data..."
    
    # Backup Prometheus data if available
    if kubectl get deployment academic-agent-prometheus -n "$NAMESPACE" &> /dev/null; then
        info "Backing up Prometheus configuration..."
        
        PROMETHEUS_POD=$(kubectl get pods -n "$NAMESPACE" -l app=prometheus -o jsonpath='{.items[0].metadata.name}')
        
        if [[ -n "$PROMETHEUS_POD" ]]; then
            # Get Prometheus configuration
            kubectl exec -n "$NAMESPACE" "$PROMETHEUS_POD" -- cat /etc/prometheus/prometheus.yml > "$BACKUP_DIR/metrics/prometheus.yml"
            
            # Get alert rules
            kubectl exec -n "$NAMESPACE" "$PROMETHEUS_POD" -- cat /etc/prometheus/alert_rules.yml > "$BACKUP_DIR/metrics/alert_rules.yml" 2>/dev/null || true
            
            # Get recording rules
            kubectl exec -n "$NAMESPACE" "$PROMETHEUS_POD" -- cat /etc/prometheus/recording_rules.yml > "$BACKUP_DIR/metrics/recording_rules.yml" 2>/dev/null || true
        fi
    fi
    
    # Backup Grafana dashboards if available
    if kubectl get deployment academic-agent-grafana -n "$NAMESPACE" &> /dev/null; then
        info "Backing up Grafana data..."
        
        GRAFANA_POD=$(kubectl get pods -n "$NAMESPACE" -l app=grafana -o jsonpath='{.items[0].metadata.name}')
        
        if [[ -n "$GRAFANA_POD" ]]; then
            # Create Grafana backup using its API
            kubectl exec -n "$NAMESPACE" "$GRAFANA_POD" -- sh -c "
                mkdir -p /tmp/grafana-backup
                cp -r /var/lib/grafana/dashboards /tmp/grafana-backup/ 2>/dev/null || true
                cp /etc/grafana/grafana.ini /tmp/grafana-backup/ 2>/dev/null || true
                tar -czf /tmp/grafana-backup.tar.gz -C /tmp/grafana-backup . 2>/dev/null || true
            "
            
            kubectl cp "$NAMESPACE/$GRAFANA_POD:/tmp/grafana-backup.tar.gz" "$BACKUP_DIR/metrics/grafana-backup.tar.gz" 2>/dev/null || true
        fi
    fi
    
    log "Monitoring backup completed"
}

# Function to create backup manifest
create_manifest() {
    log "Creating backup manifest..."
    
    cat > "$BACKUP_DIR/backup_manifest.json" << EOF
{
  "backup_id": "academic-agent-$TIMESTAMP",
  "timestamp": "$TIMESTAMP",
  "date": "$(date -Iseconds)",
  "namespace": "$NAMESPACE",
  "components": {
    "database": $([ -f "$BACKUP_DIR/database/academic_agent_prod.dump" ] && echo "true" || echo "false"),
    "redis": $([ -f "$BACKUP_DIR/redis/dump.rdb" ] && echo "true" || echo "false"),
    "logs": $([ -d "$BACKUP_DIR/logs" ] && echo "true" || echo "false"),
    "config": $([ -f "$BACKUP_DIR/config/all_resources.yaml" ] && echo "true" || echo "false"),
    "monitoring": $([ -d "$BACKUP_DIR/metrics" ] && echo "true" || echo "false")
  },
  "size_bytes": $(du -sb "$BACKUP_DIR" | cut -f1),
  "files": $(find "$BACKUP_DIR" -type f | wc -l)
}
EOF
    
    # Create human-readable summary
    cat > "$BACKUP_DIR/README.md" << EOF
# Academic Agent Backup - $TIMESTAMP

## Backup Information
- **Backup ID**: academic-agent-$TIMESTAMP
- **Date**: $(date)
- **Namespace**: $NAMESPACE
- **Total Size**: $(du -sh "$BACKUP_DIR" | cut -f1)
- **Total Files**: $(find "$BACKUP_DIR" -type f | wc -l)

## Contents

### Database Backup
$([ -f "$BACKUP_DIR/database/academic_agent_prod.dump" ] && echo "✅ PostgreSQL dump (custom format)" || echo "❌ PostgreSQL dump not found")
$([ -f "$BACKUP_DIR/database/academic_agent_prod.sql" ] && echo "✅ PostgreSQL dump (SQL format)" || echo "❌ PostgreSQL SQL dump not found")

### Redis Backup
$([ -f "$BACKUP_DIR/redis/dump.rdb" ] && echo "✅ Redis RDB dump" || echo "❌ Redis dump not found")
$([ -f "$BACKUP_DIR/redis/appendonly.aof" ] && echo "✅ Redis AOF file" || echo "❌ Redis AOF not found")

### Application Logs
$([ -d "$BACKUP_DIR/logs" ] && echo "✅ Application logs" || echo "❌ Application logs not found")

### Configuration
$([ -f "$BACKUP_DIR/config/all_resources.yaml" ] && echo "✅ Kubernetes resources" || echo "❌ Kubernetes resources not found")

### Monitoring Data
$([ -d "$BACKUP_DIR/metrics" ] && echo "✅ Monitoring configuration" || echo "❌ Monitoring data not found")

## Restoration Instructions

### Database Restoration
\`\`\`bash
# Restore from custom format
kubectl exec -i deployment/academic-agent-postgres -n $NAMESPACE -- pg_restore -U academic_agent -d academic_agent_prod --clean --if-exists < database/academic_agent_prod.dump

# Or restore from SQL format
kubectl exec -i deployment/academic-agent-postgres -n $NAMESPACE -- psql -U academic_agent -d academic_agent_prod < database/academic_agent_prod.sql
\`\`\`

### Redis Restoration
\`\`\`bash
# Copy dump file to Redis pod
kubectl cp redis/dump.rdb $NAMESPACE/\$(kubectl get pods -n $NAMESPACE -l app=redis -o jsonpath='{.items[0].metadata.name}'):/data/

# Restart Redis to load the dump
kubectl rollout restart deployment/academic-agent-redis -n $NAMESPACE
\`\`\`

### Configuration Restoration
\`\`\`bash
# Apply all resources
kubectl apply -f config/all_resources.yaml
\`\`\`
EOF
    
    log "Backup manifest created"
}

# Function to compress backup
compress_backup() {
    log "Compressing backup..."
    
    cd "$BACKUP_ROOT"
    tar -czf "academic-agent-$TIMESTAMP.tar.gz" "academic-agent-$TIMESTAMP"
    
    # Calculate checksums
    sha256sum "academic-agent-$TIMESTAMP.tar.gz" > "academic-agent-$TIMESTAMP.tar.gz.sha256"
    md5sum "academic-agent-$TIMESTAMP.tar.gz" > "academic-agent-$TIMESTAMP.tar.gz.md5"
    
    # Update manifest with compressed file info
    COMPRESSED_SIZE=$(du -sb "academic-agent-$TIMESTAMP.tar.gz" | cut -f1)
    echo "  \"compressed_size_bytes\": $COMPRESSED_SIZE," >> "$BACKUP_DIR/backup_manifest.json"
    
    log "Backup compressed: academic-agent-$TIMESTAMP.tar.gz"
}

# Function to upload to S3
upload_to_s3() {
    if [[ -z "$S3_BUCKET" ]]; then
        info "S3_BUCKET not configured, skipping S3 upload"
        return 0
    fi
    
    log "Uploading backup to S3: s3://$S3_BUCKET/$S3_PREFIX/"
    
    # Upload compressed backup
    aws s3 cp "$BACKUP_ROOT/academic-agent-$TIMESTAMP.tar.gz" \
        "s3://$S3_BUCKET/$S3_PREFIX/academic-agent-$TIMESTAMP.tar.gz" \
        --storage-class STANDARD_IA
    
    # Upload checksums
    aws s3 cp "$BACKUP_ROOT/academic-agent-$TIMESTAMP.tar.gz.sha256" \
        "s3://$S3_BUCKET/$S3_PREFIX/academic-agent-$TIMESTAMP.tar.gz.sha256"
    
    aws s3 cp "$BACKUP_ROOT/academic-agent-$TIMESTAMP.tar.gz.md5" \
        "s3://$S3_BUCKET/$S3_PREFIX/academic-agent-$TIMESTAMP.tar.gz.md5"
    
    # Upload manifest separately for easy access
    aws s3 cp "$BACKUP_DIR/backup_manifest.json" \
        "s3://$S3_BUCKET/$S3_PREFIX/manifests/academic-agent-$TIMESTAMP.json"
    
    log "Upload to S3 completed"
}

# Function to cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups (older than $RETENTION_DAYS days)..."
    
    # Cleanup local backups
    find "$BACKUP_ROOT" -name "academic-agent-*.tar.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_ROOT" -name "academic-agent-*.sha256" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_ROOT" -name "academic-agent-*.md5" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_ROOT" -type d -name "academic-agent-*" -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
    
    # Cleanup S3 backups if configured
    if [[ -n "$S3_BUCKET" ]]; then
        info "Cleaning up old S3 backups..."
        
        # List and delete old backups from S3
        aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" --recursive | \
            awk '{print $4}' | \
            while read -r file; do
                # Extract date from filename
                if [[ "$file" =~ academic-agent-([0-9]{8})_ ]]; then
                    file_date="${BASH_REMATCH[1]}"
                    current_date=$(date +%Y%m%d)
                    
                    # Calculate age in days
                    age_days=$(( ($(date -d "$current_date" +%s) - $(date -d "$file_date" +%s)) / 86400 ))
                    
                    if [[ $age_days -gt $RETENTION_DAYS ]]; then
                        info "Deleting old backup: $file (age: $age_days days)"
                        aws s3 rm "s3://$S3_BUCKET/$file"
                    fi
                fi
            done
    fi
    
    log "Cleanup completed"
}

# Function to send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Send Slack notification if webhook is configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color="good"
        if [[ "$status" != "success" ]]; then
            color="danger"
        fi
        
        local payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "Academic Agent Backup $status",
            "text": "$message",
            "fields": [
                {
                    "title": "Timestamp",
                    "value": "$TIMESTAMP",
                    "short": true
                },
                {
                    "title": "Namespace",
                    "value": "$NAMESPACE",
                    "short": true
                }
            ]
        }
    ]
}
EOF
        )
        
        curl -X POST -H 'Content-type: application/json' \
            --data "$payload" \
            "$SLACK_WEBHOOK_URL" || true
    fi
    
    # Log the notification
    if [[ "$status" == "success" ]]; then
        log "$message"
    else
        error "$message"
    fi
}

# Main backup function
main() {
    log "Starting Academic Agent backup - $TIMESTAMP"
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Create backup directory
    create_backup_dir
    
    # Perform backups
    local backup_success=true
    
    backup_database || backup_success=false
    backup_redis || backup_success=false
    backup_logs || backup_success=false
    backup_config || backup_success=false
    backup_monitoring || backup_success=false
    
    # Create manifest and compress
    create_manifest
    compress_backup
    
    # Upload to S3 if configured
    upload_to_s3 || backup_success=false
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Send notification
    if [[ "$backup_success" == "true" ]]; then
        send_notification "success" "Backup completed successfully. Size: $(du -sh "$BACKUP_ROOT/academic-agent-$TIMESTAMP.tar.gz" | cut -f1)"
        log "Backup completed successfully: $BACKUP_ROOT/academic-agent-$TIMESTAMP.tar.gz"
    else
        send_notification "failed" "Backup completed with errors. Check logs for details."
        error "Backup completed with errors"
        exit 1
    fi
}

# Run main function
main "$@"