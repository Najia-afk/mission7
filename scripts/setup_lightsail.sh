#!/bin/bash
# =============================================================================
# Mission7 Lightsail Setup Script
# 
# One-shot script to install Docker and deploy the Credit Scoring API
# on an Ubuntu Lightsail instance.
#
# Usage:
#   # Option 1: Run directly from GitHub
#   curl -sSL https://raw.githubusercontent.com/YOUR_USER/mission7/main/scripts/setup_lightsail.sh | bash
#
#   # Option 2: SSH and run manually
#   ssh -i your-key.pem ubuntu@YOUR_IP
#   bash scripts/setup_lightsail.sh
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "🚀 Mission7 Credit Scoring - Lightsail Setup"
echo "=============================================="
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_URL="${REPO_URL:-https://github.com/YOUR_USER/mission7.git}"
INSTALL_DIR="/opt/mission7"
BRANCH="${BRANCH:-main}"

# =============================================================================
# 1. INSTALL DOCKER
# =============================================================================

echo "📦 Step 1: Installing Docker..."

if command -v docker &> /dev/null; then
    echo "✅ Docker already installed: $(docker --version)"
else
    # Update package index
    sudo apt-get update
    
    # Install prerequisites
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Set up stable repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    echo "✅ Docker installed successfully"
fi

# =============================================================================
# 2. INSTALL DOCKER COMPOSE (if not included with docker-compose-plugin)
# =============================================================================

echo ""
echo "📦 Step 2: Checking Docker Compose..."

if docker compose version &> /dev/null; then
    echo "✅ Docker Compose available: $(docker compose version)"
else
    # Install standalone docker-compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed"
fi

# =============================================================================
# 3. CLONE REPOSITORY
# =============================================================================

echo ""
echo "📥 Step 3: Cloning repository..."

if [ -d "$INSTALL_DIR" ]; then
    echo "Directory exists, pulling latest changes..."
    cd $INSTALL_DIR
    git fetch origin
    git checkout $BRANCH
    git pull origin $BRANCH
else
    sudo mkdir -p $INSTALL_DIR
    sudo chown $USER:$USER $INSTALL_DIR
    git clone --branch $BRANCH $REPO_URL $INSTALL_DIR
    cd $INSTALL_DIR
fi

echo "✅ Repository ready at $INSTALL_DIR"

# =============================================================================
# 4. CREATE ENVIRONMENT FILE
# =============================================================================

echo ""
echo "🔐 Step 4: Creating environment file..."

if [ ! -f "$INSTALL_DIR/.env" ]; then
    cat > $INSTALL_DIR/.env << EOF
# PostgreSQL
POSTGRES_USER=mission7
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 24)
POSTGRES_DB=credit_scoring

# Flask
FLASK_SECRET_KEY=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
FLASK_ENV=production

# Database mode
USE_POSTGRES=true
EOF
    echo "✅ Environment file created"
    echo "⚠️  IMPORTANT: Save your .env file contents securely!"
else
    echo "✅ Environment file already exists"
fi

# =============================================================================
# 5. CREATE DATASET DIRECTORY
# =============================================================================

echo ""
echo "📁 Step 5: Preparing dataset directory..."

mkdir -p $INSTALL_DIR/dataset
mkdir -p $INSTALL_DIR/mlruns
mkdir -p $INSTALL_DIR/prod_models

echo "✅ Directories created"
echo "⚠️  Remember to copy your dataset CSVs to $INSTALL_DIR/dataset/"

# =============================================================================
# 6. BUILD AND START CONTAINERS
# =============================================================================

echo ""
echo "🐳 Step 6: Building and starting containers..."

cd $INSTALL_DIR

# Use docker compose (v2) or docker-compose (v1)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Build images
$COMPOSE_CMD -f docker-compose.prod.yml build

# Start services
$COMPOSE_CMD -f docker-compose.prod.yml up -d

echo "✅ Containers started"

# =============================================================================
# 7. WAIT FOR SERVICES
# =============================================================================

echo ""
echo "⏳ Step 7: Waiting for services to be ready..."

# Wait for Postgres
echo "Waiting for PostgreSQL..."
sleep 10

# Check if API is responding
for i in {1..30}; do
    if curl -s http://localhost/api/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

# =============================================================================
# 8. LOAD DATA (if dataset exists)
# =============================================================================

echo ""
echo "📊 Step 8: Loading data into PostgreSQL..."

if [ -f "$INSTALL_DIR/dataset/application_train.csv" ]; then
    $COMPOSE_CMD -f docker-compose.prod.yml exec -T api python scripts/load_data.py
    echo "✅ Data loaded successfully"
else
    echo "⚠️  Dataset files not found in $INSTALL_DIR/dataset/"
    echo "   Please copy CSV files and run:"
    echo "   $COMPOSE_CMD -f docker-compose.prod.yml exec api python scripts/load_data.py"
fi

# =============================================================================
# 9. FINAL STATUS
# =============================================================================

echo ""
echo "=============================================="
echo "🎉 Setup Complete!"
echo "=============================================="
echo ""
echo "Services running:"
$COMPOSE_CMD -f docker-compose.prod.yml ps
echo ""
echo "Access points:"
echo "  🌐 Landing Page:  http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP')/"
echo "  🔮 API Endpoint:  http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP')/predict"
echo "  📊 MLflow UI:     http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP')/mlflow/"
echo "  ❤️  Health Check: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_IP')/api/health"
echo ""
echo "Useful commands:"
echo "  cd $INSTALL_DIR"
echo "  $COMPOSE_CMD -f docker-compose.prod.yml logs -f    # View logs"
echo "  $COMPOSE_CMD -f docker-compose.prod.yml restart    # Restart services"
echo "  $COMPOSE_CMD -f docker-compose.prod.yml down       # Stop services"
echo ""
echo "=============================================="
