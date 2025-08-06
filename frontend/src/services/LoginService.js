const Login = async (user) => {
    const formBody = new URLSearchParams({
        username: user.username,
        password: user.password
    })

    const response = await fetch('http://localhost:8080/backend/api/auth/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: formBody
    });
}
export { Login as login };