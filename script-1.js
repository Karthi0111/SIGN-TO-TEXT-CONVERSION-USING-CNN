const translations = {
    english: {
        title: "Sign to Text",
        convertButton: "What sign do you want to convert?",
        convertedText: "Your converted text will appear here."
    },
    tamil: {
        title: "அறைகுறியியல் கலை - உரை",
        convertButton: "நீங்கள் மாற்ற வேண்டிய அறைகுறி எது?",
        convertedText: "இங்கே உங்கள் மாற்றிய உரை தோன்றும்."
    }
};

const languageToggleElements = document.querySelectorAll('.language-option');

languageToggleElements.forEach(element => {
    element.addEventListener('click', function() {
        const selectedLanguage = this.getAttribute('data-lang');

        // Change the text based on the selected language
        document.getElementById('title').innerText = translations[selectedLanguage].title;
        document.getElementById('convert-button').innerText = translations[selectedLanguage].convertButton;
        document.getElementById('converted-text').innerText = translations[selectedLanguage].convertedText;
    });
});

document.getElementById('convert-button').addEventListener('click', async function() {
    // Simulate running your Python program here
    alert('Running your Python program...'); // Replace this with actual logic

    // Simulate receiving converted text from Python
    const convertedText = "Sample converted text"; // Replace with actual conversion logic

    // Display the converted text
    document.getElementById('converted-text').innerText = convertedText;
});
