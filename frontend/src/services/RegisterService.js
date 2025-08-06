// Register function to send registration data to the backend servlet
// The backend expects form data (application/x-www-form-urlencoded), not JSON
const Register = async (user) => {
    // Prepare form data as URL-encoded string
    const formBody = new URLSearchParams({
        username: user.username,
        password: user.password,
        email: user.email
    }).toString();

    try {
        // Send POST request to backend servlet endpoint
        const response = await fetch('http://localhost:8080/backend/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: formBody
        });

        // Try to parse JSON response (if backend returns JSON)
        const data = await response.json();
        return data;
    } catch (error) {
        // Log any network or parsing errors
        console.error('Registration failed:', error);
        throw error;
    }
};

// Export the Register function for use in your React components
export { Register as register };