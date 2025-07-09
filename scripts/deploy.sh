#!/bin/bash
# WakeyWakey Deployment Script
# Automated deployment for various platforms and environments

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION=$(python -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from wakeywakey import __version__; print(__version__)")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "██╗    ██╗ █████╗ ██╗  ██╗███████╗██╗   ██╗    ██╗    ██╗ █████╗ ██╗  ██╗███████╗██╗   ██╗"
    echo "██║    ██║██╔══██╗██║ ██╔╝██╔════╝╚██╗ ██╔╝    ██║    ██║██╔══██╗██║ ██╔╝██╔════╝╚██╗ ██╔╝"
    echo "██║ █╗ ██║███████║█████╔╝ █████╗   ╚████╔╝     ██║ █╗ ██║███████║█████╔╝ █████╗   ╚████╔╝ "
    echo "██║███╗██║██╔══██║██╔═██╗ ██╔══╝    ╚██╔╝      ██║███╗██║██╔══██║██╔═██╗ ██╔══╝    ╚██╔╝  "
    echo "╚███╔███╔╝██║  ██║██║  ██╗███████╗   ██║       ╚███╔███╔╝██║  ██║██║  ██╗███████╗   ██║   "
    echo " ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝        ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   "
    echo -e "${NC}"
    echo -e "${YELLOW}                      Deployment Script v${VERSION}${NC}"
    echo ""
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        missing_deps+=("pip")
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies and try again."
        exit 1
    fi
    
    log_success "All dependencies found"
}

# Development setup
setup_development() {
    log_info "Setting up development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install package in development mode
    log_info "Installing package in development mode..."
    pip install -e .[dev,all]
    
    # Install pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        log_info "Installing pre-commit hooks..."
        pre-commit install
    fi
    
    log_success "Development environment ready!"
    log_info "Activate with: source venv/bin/activate"
}

# Production build
build_production() {
    log_info "Building production package..."
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    rm -rf build/ dist/ *.egg-info/
    
    # Build package
    python -m build
    
    # Check package
    twine check dist/*
    
    log_success "Package built successfully!"
    log_info "Built files:"
    ls -la dist/
}

# Docker deployment
deploy_docker() {
    local target=${1:-production}
    log_info "Building Docker image for target: $target"
    
    cd "$PROJECT_ROOT"
    
    # Build Docker image
    docker build --target "$target" -t "wakeywakey:$target" .
    
    # Tag with version
    docker tag "wakeywakey:$target" "wakeywakey:$target-v$VERSION"
    
    log_success "Docker image built: wakeywakey:$target"
    
    # Optionally push to registry
    if [ "$PUSH_DOCKER" = "true" ]; then
        log_info "Pushing to Docker registry..."
        docker push "wakeywakey:$target"
        docker push "wakeywakey:$target-v$VERSION"
    fi
}

# PyPI deployment
deploy_pypi() {
    local env=${1:-test}
    log_info "Deploying to PyPI ($env)..."
    
    cd "$PROJECT_ROOT"
    
    # Build if not already built
    if [ ! -d "dist" ] || [ -z "$(ls -A dist/)" ]; then
        build_production
    fi
    
    # Upload to PyPI
    if [ "$env" = "prod" ]; then
        log_warning "Uploading to production PyPI..."
        twine upload dist/*
    else
        log_info "Uploading to test PyPI..."
        twine upload --repository testpypi dist/*
    fi
    
    log_success "Package deployed to PyPI ($env)!"
}

# Raspberry Pi deployment
deploy_raspberry_pi() {
    local target_host="$1"
    log_info "Deploying to Raspberry Pi: $target_host"
    
    if [ -z "$target_host" ]; then
        log_error "Please provide target host: ./deploy.sh rpi <host>"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    
    # Create deployment package
    local deploy_dir="wakeywakey-deploy"
    rm -rf "$deploy_dir"
    mkdir "$deploy_dir"
    
    # Copy essential files
    cp -r wakeywakey "$deploy_dir/"
    cp setup.py requirements.txt README.md "$deploy_dir/"
    
    # Create installation script
    cat > "$deploy_dir/install.sh" << 'EOF'
#!/bin/bash
set -e

echo "Installing WakeyWakey on Raspberry Pi..."

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv portaudio19-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install --upgrade pip
pip install -e .

echo "Installation complete!"
echo "Activate with: source venv/bin/activate"
echo "Test with: wakeywakey list-devices"
EOF
    
    chmod +x "$deploy_dir/install.sh"
    
    # Transfer to Raspberry Pi
    log_info "Transferring files to $target_host..."
    scp -r "$deploy_dir" "$target_host:~/"
    
    # Run installation
    log_info "Running installation on $target_host..."
    ssh "$target_host" "cd ~/$deploy_dir && ./install.sh"
    
    log_success "Deployed to Raspberry Pi: $target_host"
    
    # Cleanup
    rm -rf "$deploy_dir"
}

# Microcontroller preparation
prepare_microcontroller() {
    log_info "Preparing for microcontroller deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Create minimal package
    local micro_dir="wakeywakey-micro"
    rm -rf "$micro_dir"
    mkdir -p "$micro_dir"
    
    # Extract minimal code
    python scripts/extract_minimal.py --output "$micro_dir"
    
    # Create C++ headers
    python scripts/export_cpp.py --output "$micro_dir/include"
    
    log_success "Microcontroller files prepared in $micro_dir/"
    log_info "See $micro_dir/README.md for integration instructions"
}

# Testing
run_tests() {
    log_info "Running comprehensive tests..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Run tests
    pytest tests/ -v --cov=wakeywakey --cov-report=term-missing
    
    # Run linting
    flake8 wakeywakey tests
    
    # Type checking
    mypy wakeywakey --ignore-missing-imports
    
    log_success "All tests passed!"
}

# GitHub release
create_github_release() {
    local tag="v$VERSION"
    log_info "Creating GitHub release: $tag"
    
    cd "$PROJECT_ROOT"
    
    # Check if tag exists
    if git rev-parse "$tag" >/dev/null 2>&1; then
        log_warning "Tag $tag already exists"
    else
        # Create and push tag
        git tag -a "$tag" -m "Release $tag"
        git push origin "$tag"
    fi
    
    # Create release with GitHub CLI if available
    if command -v gh &> /dev/null; then
        gh release create "$tag" \
            --title "WakeyWakey $tag" \
            --notes "Automated release $tag" \
            dist/*
        log_success "GitHub release created!"
    else
        log_warning "GitHub CLI not found. Please create release manually."
    fi
}

# Main deployment function
main() {
    print_banner
    
    local command="$1"
    shift
    
    case "$command" in
        "dev"|"development")
            check_dependencies
            setup_development
            ;;
        "build")
            check_dependencies
            build_production
            ;;
        "test")
            run_tests
            ;;
        "docker")
            deploy_docker "$@"
            ;;
        "pypi")
            deploy_pypi "$@"
            ;;
        "rpi"|"raspberry-pi")
            deploy_raspberry_pi "$@"
            ;;
        "micro"|"microcontroller")
            prepare_microcontroller
            ;;
        "release")
            check_dependencies
            run_tests
            build_production
            create_github_release
            ;;
        "all")
            check_dependencies
            setup_development
            run_tests
            build_production
            deploy_docker production
            log_success "Full deployment pipeline completed!"
            ;;
        *)
            echo "WakeyWakey Deployment Script"
            echo ""
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  dev               Setup development environment"
            echo "  build             Build production package"
            echo "  test              Run comprehensive tests"
            echo "  docker [target]   Build Docker image (production|development|minimal|arm)"
            echo "  pypi [env]        Deploy to PyPI (test|prod)"
            echo "  rpi <host>        Deploy to Raspberry Pi"
            echo "  micro             Prepare for microcontroller deployment"
            echo "  release           Create GitHub release"
            echo "  all               Run full deployment pipeline"
            echo ""
            echo "Environment Variables:"
            echo "  PUSH_DOCKER=true  Push Docker images to registry"
            echo ""
            echo "Examples:"
            echo "  $0 dev                    # Setup development"
            echo "  $0 docker production      # Build production Docker image"
            echo "  $0 rpi pi@192.168.1.100  # Deploy to Raspberry Pi"
            echo "  $0 pypi prod              # Deploy to production PyPI"
            ;;
    esac
}

# Run main function with all arguments
main "$@" 