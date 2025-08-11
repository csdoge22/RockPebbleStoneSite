import React from 'react';

const Board = () => {
  return (
    <div className="text-white min-h-screen">
      {/* Header Section */}
      <div className="flex justify-center items-center py-4 bg-green-800">
        <input
          type="text"
          placeholder="Search..."
          className="px-4 py-2 rounded-l-md border border-r-0 border-gray-300"
        />
        <button className="px-4 py-2 bg-white text-green-900 font-bold rounded-r-md">Insert</button>
      </div>

      {/* Main Content Section */}
      <div className="flex justify-around items-start mt-8 px-4">
        {/* Rock Column */}
        <div className="w-1/3 bg-gray-200 p-4 rounded-lg shadow-md">
          <h2 className="text-xl mb-2">Rock</h2>
          <ul>
            {Array.from({ length: 5 }, (_, index) => (
              <li key={index} className="flex justify-between items-center my-2 bg-gray-300 p-2 rounded">
                Item {index + 1}
                <span className="text-red-600 cursor-pointer">x</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Pebble Column */}
        <div className="w-1/3 bg-gray-200 p-4 rounded-lg shadow-md border border-r-0 border-gray-300">
          <h2 className="text-xl mb-2">Pebble</h2>
          <ul>
            {Array.from({ length: 5 }, (_, index) => (
              <li key={index} className="flex justify-between items-center my-2 bg-gray-300 p-2 rounded">
                Item {index + 1}
                <span className="text-red-600 cursor-pointer">x</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Sand Column */}
        <div className="w-1/3 bg-gray-200 p-4 rounded-lg shadow-md">
          <h2 className="text-xl mb-2">Sand</h2>
          <ul>
            {Array.from({ length: 5 }, (_, index) => (
              <li key={index} className="flex justify-between items-center my-2 bg-gray-300 p-2 rounded">
                Item {index + 1}
                <span className="text-red-600 cursor-pointer">x</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Board;