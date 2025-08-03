/** This is the home page. It is meant to showcase a brief introduction of the website */
const Home = () => {
    return (
        <>
            <div className="flex flex-col items-center justify-center min-h-screen bg-opacity-60">
                <p className="text-white text-2xl"><b>Overwhelmed by Tasks? Tackle the Rocks First.</b></p>
                <p className="text-white">Use this simple Rock, Pebble, and Sand method to cut through the noise and stay on track.</p>
            </div>
            <div className="flex flex-col items-center justify-center min-h-screen bg-opacity-60">
                <a className="bg-green-700 text-white p-2 rounded">Get Started</a>
            </div>
        </>
    )
}

export default Home;