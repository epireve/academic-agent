#!/bin/bash

# Production Deployment Script for Academic Agent
# Usage: ./deploy.sh [staging|production] [image_tag]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

# Default values
ENVIRONMENT="${1:-staging}"
IMAGE_TAG="${2:-latest}"
NAMESPACE="academic-agent"
DRY_RUN="${DRY_RUN:-false}"
TIMEOUT="${TIMEOUT:-600}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if required environment variables are set
    if [[ "$ENVIRONMENT" == "production" ]]; then
        required_vars=("AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "SECRET_KEY" "POSTGRES_PASSWORD")
        for var in "${required_vars[@]}"; do
            if [[ -z "${!var:-}" ]]; then
                error "Required environment variable $var is not set"
                exit 1
            fi
        done
    fi
    
    log "Prerequisites check passed"
}

# Function to validate configuration
validate_config() {
    log "Validating configuration..."
    
    # Check if Kubernetes manifests exist
    required_files=(
        "$DEPLOYMENT_DIR/kubernetes/namespace.yaml"
        "$DEPLOYMENT_DIR/kubernetes/secrets.yaml"
        "$DEPLOYMENT_DIR/kubernetes/postgres.yaml"
        "$DEPLOYMENT_DIR/kubernetes/redis.yaml"
        "$DEPLOYMENT_DIR/kubernetes/app.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Validate Kubernetes manifests
    for file in "${required_files[@]}"; do
        if ! kubectl apply --dry-run=client -f "$file" &> /dev/null; then
            error "Invalid Kubernetes manifest: $file"
            exit 1
        fi
    done
    
    log "Configuration validation passed"
}

# Function to create backup
create_backup() {
    if [[ "$ENVIRONMENT" != "production" ]]; then
        return 0
    fi
    
    log "Creating backup before deployment..."
    
    # Create backup directory
    BACKUP_DIR="/tmp/academic-agent-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    if kubectl get deployment academic-agent-postgres -n "$NAMESPACE" &> /dev/null; then
        info "Backing up PostgreSQL database..."
        kubectl exec -n "$NAMESPACE" deployment/academic-agent-postgres -- pg_dump -U academic_agent academic_agent_prod > "$BACKUP_DIR/postgres-backup.sql"
    fi
    
    # Backup Redis data
    if kubectl get deployment academic-agent-redis -n "$NAMESPACE" &> /dev/null; then
        info "Backing up Redis data..."
        kubectl exec -n "$NAMESPACE" deployment/academic-agent-redis -- redis-cli BGSAVE
        sleep 5
        kubectl cp "$NAMESPACE/$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}'):/data/dump.rdb" "$BACKUP_DIR/redis-backup.rdb"
    fi
    
    # Backup application logs
    if kubectl get deployment academic-agent-app -n "$NAMESPACE" &> /dev/null; then
        info "Backing up application logs..."
        kubectl logs -n "$NAMESPACE" deployment/academic-agent-app --all-containers=true > "$BACKUP_DIR/app-logs.txt"
    fi
    
    # Upload to S3 if configured
    if [[ -n "${S3_BACKUP_BUCKET:-}" ]]; then
        info "Uploading backup to S3..."
        tar -czf "$BACKUP_DIR.tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"
        aws s3 cp "$BACKUP_DIR.tar.gz" "s3://$S3_BACKUP_BUCKET/deployments/"
        rm -f "$BACKUP_DIR.tar.gz"
    fi
    
    log "Backup completed: $BACKUP_DIR"
}

# Function to update image tags
update_image_tags() {
    log "Updating image tags to: $IMAGE_TAG"
    
    # Create temporary directory for modified manifests
    TEMP_DIR=$(mktemp -d)
    cp -r "$DEPLOYMENT_DIR/kubernetes"/* "$TEMP_DIR/"
    
    # Update image tag in app.yaml
    sed -i.bak "s|image: academic-agent:latest|image: academic-agent:$IMAGE_TAG|g" "$TEMP_DIR/app.yaml"
    
    # Update any other references to the image
    find "$TEMP_DIR" -name "*.yaml" -exec sed -i.bak "s|academic-agent:latest|academic-agent:$IMAGE_TAG|g" {} \;
    
    # Remove backup files
    find "$TEMP_DIR" -name "*.bak" -delete
    
    echo "$TEMP_DIR"
}

# Function to deploy to Kubernetes
deploy_to_kubernetes() {
    local temp_dir="$1"
    
    log "Deploying to Kubernetes cluster..."
    
    # Apply namespace first
    info "Creating namespace..."
    kubectl apply -f "$temp_dir/namespace.yaml"
    
    # Apply secrets and configmaps
    info "Applying secrets and configuration..."
    kubectl apply -f "$temp_dir/secrets.yaml"
    
    # Deploy database
    info "Deploying PostgreSQL..."
    kubectl apply -f "$temp_dir/postgres.yaml"
    
    # Wait for PostgreSQL to be ready
    info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    
    # Deploy Redis
    info "Deploying Redis..."
    kubectl apply -f "$temp_dir/redis.yaml"
    
    # Wait for Redis to be ready
    info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    
    # Deploy application
    info "Deploying application..."
    kubectl apply -f "$temp_dir/app.yaml"
    
    # Wait for application deployment
    info "Waiting for application deployment to complete..."
    kubectl rollout status deployment/academic-agent-app -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    
    log "Deployment completed successfully"
}

# Function to run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Wait for pods to be ready
    info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=academic-agent -n "$NAMESPACE" --timeout=300s
    
    # Get application service
    SERVICE_NAME="academic-agent-service"
    
    # Port forward for health check (if no external access)
    info "Setting up port forwarding for health checks..."
    kubectl port-forward -n "$NAMESPACE" service/"$SERVICE_NAME" 8080:80 &
    PORT_FORWARD_PID=$!
    
    # Give port forwarding time to establish
    sleep 5
    
    # Function to cleanup port forwarding
    cleanup_port_forward() {
        if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
            kill $PORT_FORWARD_PID 2>/dev/null || true
        fi
    }
    
    # Set trap to cleanup on exit
    trap cleanup_port_forward EXIT
    
    # Run health checks
    local health_check_passed=true
    
    # Basic health check
    if ! curl -f http://localhost:8080/health --max-time 30; then
        error "Health check failed"
        health_check_passed=false
    fi
    
    # Ready check
    if ! curl -f http://localhost:8080/ready --max-time 30; then
        error "Ready check failed"
        health_check_passed=false
    fi
    
    # Metrics check
    if ! curl -f http://localhost:8080/metrics --max-time 30; then
        warn "Metrics endpoint not accessible (non-critical)"
    fi
    
    cleanup_port_forward
    
    if [[ "$health_check_passed" == "true" ]]; then
        log "All health checks passed"
    else
        error "Health checks failed"
        return 1
    fi
}

# Function to rollback deployment
rollback_deployment() {
    error "Deployment failed, initiating rollback..."
    
    # Rollback application deployment
    if kubectl get deployment academic-agent-app -n "$NAMESPACE" &> /dev/null; then
        kubectl rollout undo deployment/academic-agent-app -n "$NAMESPACE"
        kubectl rollout status deployment/academic-agent-app -n "$NAMESPACE" --timeout=300s
    fi
    
    # Restore database backup if in production
    if [[ "$ENVIRONMENT" == "production" && -n "${BACKUP_DIR:-}" ]]; then
        if [[ -f "$BACKUP_DIR/postgres-backup.sql" ]]; then
            warn "To restore database, run: kubectl exec -n $NAMESPACE deployment/academic-agent-postgres -- psql -U academic_agent academic_agent_prod < $BACKUP_DIR/postgres-backup.sql"
        fi
    fi
    
    error "Rollback completed"
}

# Function to display deployment status
show_status() {
    log "Deployment Status:"
    
    echo ""
    info "Namespace: $NAMESPACE"
    kubectl get all -n "$NAMESPACE"
    
    echo ""
    info "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    info "Service Status:"
    kubectl get services -n "$NAMESPACE"
    
    echo ""
    info "Ingress Status:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "No ingress found"
    
    echo ""
    info "Recent Events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

# Main deployment function
main() {
    log "Starting Academic Agent deployment to $ENVIRONMENT"
    log "Image tag: $IMAGE_TAG"
    
    # Check if this is a dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        warn "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Run prerequisite checks
    check_prerequisites
    validate_config
    
    # Create backup for production
    create_backup
    
    # Update image tags
    temp_dir=$(update_image_tags)
    
    # Set trap for cleanup
    cleanup() {
        rm -rf "$temp_dir"
    }
    trap cleanup EXIT
    
    # Deploy to Kubernetes
    if [[ "$DRY_RUN" != "true" ]]; then
        if ! deploy_to_kubernetes "$temp_dir"; then
            rollback_deployment
            exit 1
        fi
        
        # Run health checks
        if ! run_health_checks; then
            rollback_deployment
            exit 1
        fi
        
        # Show final status
        show_status
        
        log "Deployment to $ENVIRONMENT completed successfully!"
        
        # Output useful information
        echo ""
        info "Useful commands:"
        echo "  Monitor pods: kubectl get pods -n $NAMESPACE -w"
        echo "  View logs: kubectl logs -f deployment/academic-agent-app -n $NAMESPACE"
        echo "  Port forward: kubectl port-forward -n $NAMESPACE service/academic-agent-service 8080:80"
        echo "  Delete deployment: kubectl delete namespace $NAMESPACE"
        
    else
        log "DRY RUN completed - no changes made"
    fi
}

# Script usage
usage() {
    echo "Usage: $0 [staging|production] [image_tag]"
    echo ""
    echo "Options:"
    echo "  staging|production  Target environment (default: staging)"
    echo "  image_tag           Docker image tag to deploy (default: latest)"
    echo ""
    echo "Environment variables:"
    echo "  DRY_RUN            Set to 'true' for dry run (default: false)"
    echo "  TIMEOUT            Deployment timeout in seconds (default: 600)"
    echo "  S3_BACKUP_BUCKET   S3 bucket for backups (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 staging"
    echo "  $0 production v1.2.3"
    echo "  DRY_RUN=true $0 production latest"
}

# Parse command line arguments
if [[ "$#" -gt 2 ]]; then
    usage
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    usage
    exit 1
fi

# Run main function
main "$@"