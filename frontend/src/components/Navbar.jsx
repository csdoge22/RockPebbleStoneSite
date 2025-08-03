const Navbar = () => {
    /* If user is not logged in*/
    return (
        <nav className="navbar flex bg-green-700 p-4">
            <ul className="flex space-x-4">
                <li className="nav-item bg-green-700 active:bg-green-500"><a href="/">Home</a></li>
                <li className="nav-item active:bg-green-500"><a href="/login">Login</a></li>
                <li className="nav-item active:bg-green-500"><a href="/register">Register</a></li>
                <li className="nav-item active:bg-green-500"><a href="/about">About</a></li>
                <li className="nav-item active:bg-green-500"><a href="/board">Board</a></li>
            </ul>
        </nav>
    )
}
export default Navbar;