/** This is the about page. It provides detailed information about the website, its purpose, and more. */
const About = () => {
    return (
        <div>
            <div>
                <h1 className="text-white text-4xl">About AI Sand Management Solutions</h1>
                <p className="text-white">This website is designed to help users prioritize their tasks using the Rock, Pebble, and Sand method.</p>
                <p className="text-white">It allows users to categorize their tasks into three levels of priority, making it easier to focus on what matters most.</p>
                <p className="text-white">Users can add, edit, and delete tasks, and view them in a user-friendly interface.</p>
                <p className="text-white">Whether you're overwhelmed with tasks or just looking for a better way to manage your time, this site aims to provide a simple and effective solution.</p>
            </div>
            <div>
                <h2 className="text-white text-3xl mt-8">How to use the site</h2>
                <p className="text-white">1. Register an account to start managing your tasks.</p>
                <p className="text-white">2. Log in to access your task board.</p>
                <p className="text-white">3. Add tasks and an AI will classify them as Rocks, Pebbles, or Sand.</p>
                <p className="text-white">4. View your tasks and prioritize them based on their importance.</p>
                <p className="text-white">5. Edit or delete tasks as needed to keep your board up-to-date.</p>
            </div>
            <div>
                <h1 className="text-white text-4xl mt-8">Frequently Asked Questions</h1>
                <h2 className="text-white text-3xl mt-4">Q: How does the AI classification work?</h2>
                <p className="text-white">A: The AI analyzes your task descriptions and categorizes them based on their content and urgency.</p>
                <h2 className="text-white text-3xl mt-4">Q: Can I customize the categories?</h2>
                <p className="text-white">A: Currently, the categories are fixed as Rocks, Pebbles, and Sand, but we may add customization options in the future.</p>
            </div>
        </div>
    )
}
export default About;