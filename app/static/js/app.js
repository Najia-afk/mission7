/**
 * Credit Scoring Application - Shared JavaScript
 * Prêt à dépenser - MLOps Platform
 */

// Mobile menu toggle
function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    if (navLinks) {
        navLinks.classList.toggle('active');
    }
}

// Close mobile menu when clicking outside
document.addEventListener('click', (e) => {
    const navLinks = document.querySelector('.nav-links');
    const menuToggle = document.querySelector('.menu-toggle');
    
    if (navLinks && navLinks.classList.contains('active')) {
        if (!navLinks.contains(e.target) && !menuToggle.contains(e.target)) {
            navLinks.classList.remove('active');
        }
    }
});

// Format numbers with locale
function formatNumber(num, decimals = 0) {
    if (num === null || num === undefined) return '-';
    return new Intl.NumberFormat('fr-FR', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(num);
}

// Format currency
function formatCurrency(num) {
    if (num === null || num === undefined) return '-';
    return new Intl.NumberFormat('fr-FR', {
        style: 'currency',
        currency: 'EUR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(num);
}

// Format percentage
function formatPercent(num, decimals = 1) {
    if (num === null || num === undefined) return '-';
    return (num * 100).toFixed(decimals) + '%';
}

// Show loading state on button
function setButtonLoading(btn, loading, originalText = null) {
    if (loading) {
        btn.dataset.originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Loading...';
    } else {
        btn.disabled = false;
        btn.innerHTML = originalText || btn.dataset.originalText || 'Submit';
    }
}

// Show toast notification
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    // Create toast container if not exists
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(toast);
    
    // Auto remove
    setTimeout(() => {
        toast.classList.add('toast-fade-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Execute SHAP HTML scripts
function executeShapScripts(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const scripts = container.getElementsByTagName('script');
    for (let script of scripts) {
        const newScript = document.createElement('script');
        newScript.text = script.innerHTML;
        document.body.appendChild(newScript).parentNode.removeChild(newScript);
    }
}

// API helper
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    if (data && method !== 'GET') {
        options.body = JSON.stringify(data);
    }
    
    try {
        const response = await fetch(endpoint, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || `HTTP ${response.status}`);
        }
        
        return result;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Smooth scroll to element
function scrollToElement(elementId, offset = 100) {
    const element = document.getElementById(elementId);
    if (element) {
        const top = element.getBoundingClientRect().top + window.pageYOffset - offset;
        window.scrollTo({ top, behavior: 'smooth' });
    }
}

// Copy to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success');
    } catch (err) {
        console.error('Copy failed:', err);
        showToast('Failed to copy', 'error');
    }
}

// Check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Add toast styles dynamically
const toastStyles = document.createElement('style');
toastStyles.textContent = `
    .toast-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 10000;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    .toast {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 16px;
        border-radius: 8px;
        background: var(--bg-secondary, #1e1e1e);
        border: 1px solid var(--border-color, #333);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        animation: toastIn 0.3s ease;
    }
    
    .toast-info { border-left: 4px solid var(--primary, #3b82f6); }
    .toast-success { border-left: 4px solid var(--success, #22c55e); }
    .toast-warning { border-left: 4px solid var(--warning, #f59e0b); }
    .toast-error { border-left: 4px solid var(--danger, #ef4444); }
    
    .toast-close {
        background: none;
        border: none;
        color: var(--text-muted, #888);
        cursor: pointer;
        font-size: 18px;
        padding: 0 4px;
    }
    
    .toast-fade-out {
        animation: toastOut 0.3s ease forwards;
    }
    
    @keyframes toastIn {
        from { opacity: 0; transform: translateX(100%); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes toastOut {
        from { opacity: 1; transform: translateX(0); }
        to { opacity: 0; transform: translateX(100%); }
    }
`;
document.head.appendChild(toastStyles);

// Export for modules if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        toggleMobileMenu,
        formatNumber,
        formatCurrency,
        formatPercent,
        setButtonLoading,
        showToast,
        executeShapScripts,
        apiCall,
        debounce,
        scrollToElement,
        copyToClipboard,
        isInViewport
    };
}
