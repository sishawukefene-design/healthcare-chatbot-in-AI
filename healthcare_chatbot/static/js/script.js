class HealthcareChatbot {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.messagesArea = document.getElementById('messagesArea');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        this.themeToggle = document.getElementById('themeToggle');
        this.exportBtn = document.getElementById('exportBtn');
        this.suggestionsBar = document.getElementById('suggestionsBar');
        this.isProcessing = false;
        this.messageCount = 0;
        this.init();
    }
    init() {
        // Event listeners
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.newChatBtn.addEventListener('click', () => this.newChat());
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        this.themeToggle.addEventListener('click', () => this.toggleTheme());
        this.exportBtn.addEventListener('click', () => this.exportChat());
        this.messageInput.addEventListener('input', () => this.autoResize());
        this.loadSuggestions();
        this.messageInput.focus();
        this.updateStats();
    }
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;
        this.messageInput.value = '';
        this.autoResize();
        this.addMessage(message, 'user');
        this.messageCount++;
        this.showTyping();
        this.isProcessing = true;
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            const data = await response.json();
            if (response.ok) {
                this.addBotMessage(data);
                this.updateSuggestions(data.suggestions);
            } else {
                this.addMessage('Sorry, an error occurred. Please try again.', 'bot', true);
            }
        } catch (error) {
            console.error('Error:', error);
            this.addMessage('Network error. Please check your connection.', 'bot', true);
        } finally {
            this.hideTyping();
            this.isProcessing = false;
            this.updateStats();
        }
    }
    addMessage(text, sender, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${sender === 'user' ? 'fa-user' : 'fa-robot'}"></i>
            </div>
            <div class="message-content" style="${isError ? 'background: #fc8181; color: white;' : ''}">
                <div class="message-header">
                    <span class="sender-name">${sender === 'user' ? 'You' : 'HealthAI'}</span>
                    <span class="message-time">${timeStr}</span>
                </div>
                <div class="message-text">${this.formatMessage(text)}</div>
            </div>
        `;
        this.messagesArea.appendChild(messageDiv);
        this.scrollToBottom();
    }
    addBotMessage(data) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Confidence badge
        const confidenceColor = this.getConfidenceColor(data.confidence);

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">HealthAI</span>
                    <span class="message-time">${timeStr}</span>
                </div>
                <div class="message-text">${this.formatMessage(data.answer)}</div>
                <div class="message-footer" style="margin-top: 10px; font-size: 11px; opacity: 0.7;">
                    <span>📚 Source: ${data.topic}</span>
                    <span style="margin-left: 10px;">📊 Confidence:
                        <span style="color: ${confidenceColor}">${data.confidence} (${data.score}%)</span>
                    </span>
                </div>
            </div>
        `;

        this.messagesArea.appendChild(messageDiv);
        this.scrollToBottom();
    }
    formatMessage(text) {
        // Convert markdown-like syntax to HTML
        let formatted = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/•/g, '<br>•');

        // Add bullet points formatting
        formatted = formatted.replace(/<br>•/g, '<br>&nbsp;&nbsp;•');

        return formatted;
    }
    getConfidenceColor(confidence) {
        switch(confidence) {
            case 'Very High': return '#48bb78';
            case 'High': return '#68d391';
            case 'Medium': return '#ed8936';
            case 'Low': return '#fc8181';
            default: return '#a0aec0';
        }
    }

    async loadSuggestions() {
        try {
            const response = await fetch('/api/suggestions');
            const data = await response.json();
            this.updateSuggestions(data.suggestions);
        } catch (error) {
            console.error('Failed to load suggestions:', error);
        }
    }

    updateSuggestions(suggestions) {
        if (!suggestions || suggestions.length === 0) {
            this.suggestionsBar.style.display = 'none';
            return;
        }
        this.suggestionsBar.style.display = 'flex';
        this.suggestionsBar.innerHTML = suggestions.map(suggestion =>
            `<span class="suggestion-chip" onclick="chatbot.useSuggestion('${suggestion.replace(/'/g, "\\'")}')">
                ${suggestion}
            </span>`
        ).join('');
    }

    useSuggestion(suggestion) {
        this.messageInput.value = suggestion;
        this.sendMessage();
    }

    showTyping() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }

    hideTyping() {
        this.typingIndicator.style.display = 'none';
    }

    scrollToBottom() {
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    autoResize() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    async newChat() {
        if (confirm('Start a new chat? Current conversation will be saved.')) {
            this.messagesArea.innerHTML = '';
            this.messageCount = 0;
            await this.loadWelcomeMessage();
            this.messageInput.value = '';
            this.messageInput.focus();
        }
    }

    async clearHistory() {
        if (confirm('Clear all chat history? This cannot be undone.')) {
            try {
                await fetch('/api/clear', { method: 'POST' });
                this.messagesArea.innerHTML = '';
                await this.loadWelcomeMessage();
                this.messageCount = 0;
                this.updateStats();
            } catch (error) {
                console.error('Failed to clear history:', error);
            }
        }
    }

    async loadWelcomeMessage() {
        // Reload welcome message
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message bot-message welcome-message';
        welcomeDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">HealthAI Assistant</span>
                </div>
                <div class="message-text">
                    <h3>👋 Welcome back!</h3>
                    <p>How can I help you with your health questions today?</p>
                </div>
            </div>
        `;
        this.messagesArea.appendChild(welcomeDiv);
        this.scrollToBottom();
    }

    toggleTheme() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');

        if (currentTheme === 'dark') {
            html.removeAttribute('data-theme');
            this.themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        } else {
            html.setAttribute('data-theme', 'dark');
            this.themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        }

        // Save preference
        localStorage.setItem('theme', currentTheme === 'dark' ? 'light' : 'dark');
    }

    exportChat() {
        const messages = document.querySelectorAll('.message');
        let exportText = 'Healthcare Chatbot Conversation\n';
        exportText += '='.repeat(50) + '\n\n';

        messages.forEach(msg => {
            const sender = msg.querySelector('.sender-name')?.innerText || '';
            const text = msg.querySelector('.message-text')?.innerText || '';
            const time = msg.querySelector('.message-time')?.innerText || '';

            if (sender && text) {
                exportText += `[${time}] ${sender}:\n${text}\n\n`;
            }
        });

        const blob = new Blob([exportText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_export_${new Date().toISOString().slice(0,19)}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }

    async updateStats() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            const chatCount = data.history.length;

            document.getElementById('chatCount').innerHTML = chatCount;
            document.getElementById('docCount').innerHTML = '20+ Topics';

            // Update history list
            const historyList = document.getElementById('historyList');
            if (historyList && data.history.length > 0) {
                historyList.innerHTML = data.history.slice(-5).reverse().map(item => `
                    <div class="history-item" onclick="chatbot.loadHistoryMessage('${item.user.replace(/'/g, "\\'")}')">
                        ${item.user.substring(0, 40)}${item.user.length > 40 ? '...' : ''}
                    </div>
                `).join('');
            }
        } catch (error) {
            console.error('Failed to update stats:', error);
        }
    }

    loadHistoryMessage(message) {
        this.messageInput.value = message;
        this.sendMessage();
    }
}

// Initialize chatbot
let chatbot;
document.addEventListener('DOMContentLoaded', () => {
    chatbot = new HealthcareChatbot();

    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        document.getElementById('themeToggle').innerHTML = '<i class="fas fa-sun"></i>';
    }
});