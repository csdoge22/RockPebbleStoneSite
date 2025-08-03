const Navbar = () => {
    /* If user is not logged in*/
    return (
        <nav className="navbar bg-green-500 p-4">
            <ul>
                <li className="nav-item"><a href="/">Home</a></li>
                <li className="nav-item"><a href="/login">Login</a></li>
                <li className="nav-item"><a href="/register">Register</a></li>
                <li className="nav-item"><a href="/about">About</a></li>
                <li className="nav-item"><a href="/board">Board</a></li>
            </ul>
        </nav>
    )
}
export default Navbar;