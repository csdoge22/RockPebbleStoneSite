const Logout = () => {
    return (
        <div className="flex flex-col items-center justify-center min-h-screen">
            <h1 className="text-3xl font-bold mb-4 text-gray-800">You have been logged out</h1>
            <p className="text-lg text-gray-600">Thank you for visiting! See you next time.</p>
            <p>Click
                <a className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600" href="/">
                    here
                </a>
                 to go back to the home page
            </p>
        </div>
    )
}
export default Logout;