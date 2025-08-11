const Login = async (user) => {
    const formBody = new URLSearchParams({
        username: user.username,
        password: user.password
    });

    try {
        const response = await fetch('http://localhost:8080/backend/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: formBody
        });

        // Handle non-2xx responses
        if (!response.ok) {
            const errorData = await response.json();
            return { 
                success: false, 
                error: errorData.error || "Login failed" 
            };
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Network error:', error);
        return { 
            success: false, 
            error: "Network error. Please try again." 
        };
    }
};

export { Login as login };