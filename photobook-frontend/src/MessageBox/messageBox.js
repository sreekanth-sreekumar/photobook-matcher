import './messageBox.css'

const MessageBox = (props) => {
    const {messageList, handleEnter} = props;
    return (
        <div className='message-box'>
            <div className="message-list">
                {messageList.map(msg => <p className={`message-${msg.user}`}>{msg.value}</p>)}
            </div>
            <div className="message-enter">
                <input 
                    className = "message-input" 
                    placeholder="Enter input here"
                    onKeyUp={e => {
                        if (e.keyCode === 13) {
                            handleEnter(e.target.value);
                            e.target.value = '';
                        }
                    }}
                />
            </div>
        </div>
    )
}

export default MessageBox;