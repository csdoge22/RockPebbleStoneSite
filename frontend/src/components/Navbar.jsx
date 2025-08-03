const Navbar = () => {
    /* If user is not logged in*/
    return (
        <nav>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/login">Login</a></li>
                <li><a href="/register">Register</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/board">Board</a></li>
            </ul>
        </nav>
    )
}
export default Navbar;