// script.js

// Elements for language buttons
const langEnButton = document.getElementById('lang-en');
const langTaButton = document.getElementById('lang-ta');
const convertBtn = document.getElementById('convert-btn');

// Track current language
let currentLanguage = 'en';

// Update text based on selected language
function updateLanguage() {
    if (currentLanguage === 'en') {
        convertBtn.textContent = "What sign do you need to convert?";
        langEnButton.style.backgroundColor = "#333";
        langTaButton.style.backgroundColor = "#444";
    } else {
        convertBtn.textContent = "நீங்கள் எவ்வித அடையாளத்தை மாற்ற விரும்புகிறீர்கள்?";
        langTaButton.style.backgroundColor = "#333";
        langEnButton.style.backgroundColor = "#444";
    }
}

// Event listeners for language buttons
langEnButton.addEventListener('click', () => {
    currentLanguage = 'en';
    updateLanguage();
});

langTaButton.addEventListener('click', () => {
    currentLanguage = 'ta';
    updateLanguage();
});

// Event listener for the convert button
convertBtn.addEventListener('click', () => {
    // Open your sign language to text conversion program
    alert("Starting sign-to-text conversion program...");
    // Integrate the actual program here if it's web-based or connected to a server.
});

// Set initial language text
updateLanguage();
