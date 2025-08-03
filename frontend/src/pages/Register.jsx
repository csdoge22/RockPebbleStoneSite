const Register = () => {
    return (
        <>
        <h1>Signup Here</h1>
        <form>
            <label htmlFor="username">Username:</label>
            <input type="text" id="username" name="username" required />
            <br />
            <label htmlFor="email">Email:</label>
            <input type="email" id="email" name="email" required />
            <br />
            <label htmlFor="password">Password:</label>
            <input type="password" id="password" name="password" required />
            <br />
            <button type="submit">Register</button>
        </form>
        </>
    )
}
export default Register;