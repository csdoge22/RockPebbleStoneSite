const Login = () => {
    /* This is still in beta phase so I will not add an option for forget password */
    return (
        <div className="flex items-center justify-center min-h-screen bg-opacity-60">
            <form className="bg-white border-4 border-blue-300 rounded-md p-8 w-full max-w-md shadow-lg">
                <h1 className="text-3xl font-bold underline text-center mb-8">HELLO THERE</h1>
                <div className="mb-6 flex items-center">
                    <label htmlFor="username" className="w-32 text-lg text-black">Username</label>
                    <input type="text" id="username" name="username" className="flex-1 border border-gray-300 bg-gray-200 px-2 py-1 rounded focus:outline-none" required />
                </div>
                <div className="mb-6 flex items-center">
                    <label htmlFor="password" className="w-32 text-lg text-black">Password</label>
                    <input type="password" id="password" name="password" className="flex-1 border border-gray-300 bg-gray-200 px-2 py-1 rounded focus:outline-none" required />
                </div>
                <div className="mb-6 text-center">
                    <span>Don't have an account yet? </span>
                    <a href="/register" className="text-blue-600 underline">Register</a>
                </div>
                <button type="submit" className="w-full bg-green-900 text-white py-2 rounded hover:bg-green-800 transition">Login</button>
            </form>
        </div>
    )
}
export default Login;