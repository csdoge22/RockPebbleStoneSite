import { useState } from "react";
import React from 'react';
const Navbar = () => {
    const [isLoggedIn, setIsLoggedIn] = React.useState(false);

    const handleLogin = () => setIsLoggedIn(true);
    const handleLogout = () => setIsLoggedIn(false);
    /* If user is not logged in*/
    return (
        <nav className="navbar flex bg-green-700 p-4">
            <ul className="flex space-x-4">
                { isLoggedIn ? (
                <>
                    <li className="nav-item bg-green-700 active:bg-green-500"><a href="/">Home</a></li>
                    <li className="nav-item active:bg-green-500"><a href="/login">Login</a></li>
                    <li className="nav-item active:bg-green-500"><a href="/register">Register</a></li>
                    <li className="nav-item active:bg-green-500"><a href="/about">About</a></li>
                    <li className="nav-item active:bg-green-500"><a href="/board">Board</a></li>
                </>
                ) : (
                    <>
                        <li className="nav-item bg-green-700 active:bg-green-500"><a href="/">Home</a></li>
                        <li className="nav-item active:bg-green-500"><a href="/login">Login</a></li>
                        <li className="nav-item active:bg-green-500"><a href="/register">Register</a></li>
                        <li className="nav-item active:bg-green-500"><a href="/about">About</a></li>
                    </>
                )}
            </ul>
        </nav>
    )
}
export default Navbar;