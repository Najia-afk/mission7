#!/bin/bash
# =============================================================================
# Mission7 - Initial Server Setup Script
# =============================================================================
#
# Run this ONCE on a fresh Ubuntu server to set up everything.
# After this, CI/CD will use the deploy user to git pull and docker compose.
#
# Usage:
#   sudo ./deploy.sh <deploy_username> <deploy_password>
#
# Example:
#   sudo ./deploy.sh deploy mySecurePass123
#
# =============================================================================

set -e

# Configuration
DEPLOY_USER="${1:-deploy}"
DEPLOY_PASS="${2:-}"
INSTALL_DIR="/opt/mission7"
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/mission7.git}"
BRANCH="${BRANCH:-main}"

echo "========================================"
echo "Mission7 - Server Setup"
echo "========================================"

# Prompt for password if not provided
if [ -z "$DEPLOY_PASS" ]; then
    read -sp "Enter password for deploy user '$DEPLOY_USER': " DEPLOY_PASS
    echo
fi

if [ -z "$DEPLOY_PASS" ]; then
    echo "Error: Password is required"
    exit 1
fi

# =============================================================================
# Step 1: Install Docker
# =============================================================================

echo "[1/4] Installing Docker..."

if ! command -v docker &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq apt-transport-https ca-certificates curl gnupg lsb-release git
    
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
    
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    echo "Docker installed"
else
    echo "Docker already installed"
fi

# =============================================================================
# Step 2: Create Deploy User
# =============================================================================

echo "[2/4] Creating deploy user '$DEPLOY_USER'..."

if id "$DEPLOY_USER" &>/dev/null; then
    echo "User '$DEPLOY_USER' already exists"
else
    useradd -m -s /bin/bash "$DEPLOY_USER"
    echo "$DEPLOY_USER:$DEPLOY_PASS" | chpasswd
    echo "User '$DEPLOY_USER' created"
fi

# Add to docker group
usermod -aG docker "$DEPLOY_USER"

# =============================================================================
# Step 3: Clone Repository
# =============================================================================

echo "[3/4] Cloning repository..."

mkdir -p "$INSTALL_DIR"
chown "$DEPLOY_USER:$DEPLOY_USER" "$INSTALL_DIR"

if [ -d "$INSTALL_DIR/.git" ]; then
    cd "$INSTALL_DIR"
    sudo -u "$DEPLOY_USER" git fetch origin
    sudo -u "$DEPLOY_USER" git checkout "$BRANCH"
    sudo -u "$DEPLOY_USER" git pull origin "$BRANCH"
    echo "Repository updated"
else
    sudo -u "$DEPLOY_USER" git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    echo "Repository cloned"
fi

# =============================================================================
# Step 4: Initial Build
# =============================================================================

echo "[4/4] Building and starting services..."

cd "$INSTALL_DIR"
sudo -u "$DEPLOY_USER" docker compose -f docker-compose.prod.yml up -d --build

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Deploy user credentials (save for GitHub Secrets):"
echo "  DEPLOY_USER: $DEPLOY_USER"
echo "  DEPLOY_PASS: <the password you entered>"
echo "  DEPLOY_HOST: $(hostname -I | awk '{print $1}')"
echo ""
echo "CI/CD command:"
echo "  cd $INSTALL_DIR && git pull && docker compose -f docker-compose.prod.yml up -d --build"
echo ""
echo "Access: http://$(hostname -I | awk '{print $1}')"
echo ""
