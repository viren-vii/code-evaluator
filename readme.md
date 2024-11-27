# Code evaluator

This is a simple code evaluator that uses a LLM to evaluate the correctness of a code snippet.

## How to use

1. Clone this repository
2. Run `pip install -r requirements.txt`
3. Add `.env` file with the following variables:
    - `OPENAI_API_KEY`: Your OpenAI API key
4. Run `python app.py`
5. The server will run on `localhost:8000`
6. Clone the [client repository](https://github.com/viren-vii/code-evaluator-client) and follow the instructions to run the client.

## How it works

- The evaluator uses a LLM to evaluate the correctness of a code snippet. It uses the `evaluate` endpoint of the server.
- The evaluator uses the `messages` field to get the conversation history between the user and the LLM.
- The evaluator uses the `user_code_snippet` field to get the code snippet that the user has written.
- The evaluator uses the `thread_id` field to get the thread id.
- The evaluator returns a `ModelResponse` object.

## API Endpoints

### 1. Initialize Conversation
- **Endpoint**: `POST /init/`
- **Description**: Initializes a new conversation and returns a thread ID
- **Request**: No payload required
- **Response**:
  ```json
  {
    "messages": Array<Message>,
    "final_output": {
      "response": string
    },
    "thread_id": string
  }
  ```

### Message Types
Messages in the array can be of three types:
- System Message: `{"type": "system", "content": string}`
- Human Message: `{"type": "human", "content": string}`
- AI Message: `{"type": "ai", "content": string}`

### 2. Evaluate Code
- **Endpoint**: `POST /evaluate/`
- **Description**: Evaluates a code snippet and provides feedback
- **Request Body**:
  ```json
  {
    "messages": Array<Message>,
    "user_code_snippet": string,
    "thread_id": string
  }
  ```
- **Response**:
  ```json
    {
        "messages": Array<Message>,
        "final_output": {
            "status": string,
            "score": number,
            "comment": string,
            "hint": string
        },
        "thread_id": string
    }
  ```
### 3. Feed Question
- **Endpoint**: `POST /feed-question/`
- **Description**: Feeds a new question into the conversation
- **Request Body**:
  ```json
  {
    "question": string,
    "messages": Array<Message>,
    "thread_id": string
  }
  ```
- **Response**:
  ```json
  {
    "messages": Array<Message>,
    "final_output": {
      "response": string
    },
    "thread_id": string
  }
  ```

## Demo video

<video width="640" height="360" controls>
  <source src="https://github.com/viren-vii/code-evaluator/blob/main/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>