import './App.css';
import ImageGrid from './ImageGrid/imageGrid';
import MessageBox from './MessageBox/messageBox';
import { useState } from 'react';

function App() {
  const [messageList, setMessageList] = useState([])
  const [numRounds, setNumRounds] = useState(1);
  const [endState, setEndStateClicked] = useState(false);
  const handleEnter = (message) => {
    const newList = messageList.concat({user: 'human', value: message});
    setMessageList(newList)
    fetch(`/get_answer/${message}`)
    .then(res => res.text())
    .then(data => {
      setTimeout(function() { setMessageList(newList.concat({user: 'system', value: data})) }, 2000)
    })
  }

  if (endState) {
    return <p className="end-tag">You have played a total of {numRounds}</p>
  }

  return (
    <div className="App">
      <header className="app-header">
        Photobook Matcher
      </header>
      <div className="main-layout">
        <ImageGrid roundNr={numRounds}/>
        <MessageBox messageList={messageList} handleEnter={handleEnter}/>
      </div>
      <div className="footer-actions">
        <button 
          className="footer-button"
          onClick={() => {
            setMessageList([])
            setNumRounds(numRounds + 1)
          }}  
        >Next</button>
        <button 
          className="footer-button"
          onClick={() => setEndStateClicked(true)}
        >End</button>
      </div>
    </div>
  )
}

export default App;
