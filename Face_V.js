const toggleFormButton = document.getElementById('toggle-form');
const loginForm = document.getElementById('login-form');
const signupForm = document.getElementById('signup-form');

toggleFormButton.addEventListener('click', () => {
    loginForm.classList.toggle('hidden');
    signupForm.classList.toggle('hidden');
    toggleFormButton.textContent = 
        toggleFormButton.textContent === 'Switch to Login' ? 'Switch to Sign Up' : 'Switch to Login';
});

