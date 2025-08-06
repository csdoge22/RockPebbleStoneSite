import { useState } from "react";
import { register } from "../services/RegisterService";
import React from "react";

const Register = () => {
    const [form, setForm] = useState({ username: "", password: "", email: "" });
    const [error, setError] = useState("");

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const response = await register(form);
        console.log("Backend response:", response);
        if(response && response.success){
            window.location.href = "/login";
        } else {
            setError(response && response.error ? response.error : "Unknown error");
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-opacity-60">
            <form className="bg-white border-4 border-blue-300 rounded-md p-8 w-full max-w-md shadow-lg" onSubmit={handleSubmit}>
                <h1 className="text-3xl font-bold underline text-center mb-8">HELLO THERE</h1>
                <div className="mb-6 flex items-center">
                    <label htmlFor="username" className="w-32 text-lg text-black">Username</label>
                    <input type="text" id="username" name="username" value={form.username} onChange={handleChange} className="flex-1 border border-gray-300 bg-gray-200 px-2 py-1 rounded focus:outline-none" required />
                </div>
                <div className="mb-6 flex items-center">
                    <label htmlFor="password" className="w-32 text-lg text-black">Password</label>
                    <input type="password" id="password" name="password" value={form.password} onChange={handleChange} className="flex-1 border border-gray-300 bg-gray-200 px-2 py-1 rounded focus:outline-none" required />
                </div>
                <div className="mb-6 flex items-center">
                    <label htmlFor="email" className="w-32 text-lg text-black">Email</label>
                    <input type="email" id="email" name="email" value={form.email} onChange={handleChange} className="flex-1 border border-gray-300 bg-gray-200 px-2 py-1 rounded focus:outline-none" required />
                </div>
                <div className="mb-6 text-center">
                    <span>Already have an account? </span>
                    <a href="/login" className="text-blue-600 underline">Login</a>
                </div>
                <button type="submit" className="w-full bg-green-900 text-white py-2 rounded hover:bg-green-800 transition">Sign Up</button>
                {error && <p className="text-red-500 text-center mt-4">{error}</p>}
            </form>
        </div>
    )
}
export default Register;