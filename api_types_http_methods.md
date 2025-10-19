
# Types of APIs

APIs (Application Programming Interfaces) can be categorized based on their use cases, accessibility, and architecture. Common types include:

## 1. **Open APIs (Public APIs)**
- Available to external users and developers.
- Example: Twitter API, Google Maps API.

## 2. **Internal APIs (Private APIs)**
- Used within an organization.
- Not exposed to external users.

## 3. **Partner APIs**
- Shared with specific partners.
- Require specific rights or licenses.

## 4. **Composite APIs**
- Combine multiple API calls into a single call.
- Useful for microservices architectures.

## 5. **Web APIs**
- Accessed over the web using HTTP/HTTPS.
- Examples: REST, SOAP, GraphQL.

## 6. **Library/Framework APIs**
- Provided by software libraries or frameworks.
- Used within applications for specific functionalities.


## 7. **REST APIs (Representational State Transfer)**
- Follow REST architectural principles.
- Use standard HTTP methods (GET, POST, PUT, DELETE, etc.).
- Stateless communication and resource-based URLs.
- Widely used for web services due to simplicity and scalability.


## 8. **SOAP APIs (Simple Object Access Protocol)**
- Use XML-based messaging protocol for communication.
- Rely on strict standards and formal contracts (WSDL).
- Support operations over HTTP, SMTP, and other protocols.
- Common in enterprise environments requiring high security and ACID compliance.


## 9. **GraphQL APIs**
- Use a query language for APIs developed by Facebook.
- Allow clients to request exactly the data they need.
- Single endpoint handles multiple types of queries and mutations.
- Reduces over-fetching and under-fetching of data.
- Strongly typed schema defines available data and operations.


---


# HTTP Methods

HTTP methods define the actions that can be performed on resources in a web API. The most commonly used HTTP methods are:

| Method  | Description                                 | Typical Use Case           |
|---------|---------------------------------------------|----------------------------|
| GET     | Retrieve data from the server               | Fetching resources         |
| POST    | Submit data to the server                   | Creating new resources     |
| PUT     | Update existing data on the server          | Replacing resources        |
| PATCH   | Partially update existing data              | Modifying part of resource |
| DELETE  | Remove data from the server                 | Deleting resources         |
| OPTIONS | Describe communication options for resource | Discovering capabilities   |
| HEAD    | Retrieve headers for a resource             | Checking metadata          |

These methods are fundamental to RESTful API design and are used to interact with resources in a standardized way.


---

# Additional API Concepts

## **API Authentication & Authorization**
- Mechanisms to secure APIs and control access.
- Common methods: API keys, OAuth, JWT, Basic Auth.

## **API Rate Limiting**
- Restricts the number of API requests a client can make in a given time period.
- Helps prevent abuse and ensures fair usage.

## **API Documentation**
- Provides details on how to use the API, including endpoints, request/response formats, and examples.
- Tools: Swagger/OpenAPI, Postman.

## **Versioning**
- Allows APIs to evolve without breaking existing clients.
- Common strategies: URI versioning (`/v1/resource`), header versioning.

--- 

These concepts are essential for building robust, secure, and maintainable APIs.



# API Endpoints

API endpoints are specific paths or URLs exposed by an API to interact with resources. Each endpoint corresponds to a particular function or data entity.

## **Structure of an Endpoint**
- Typically follows the pattern: `https://api.example.com/resource`
- Can include parameters, such as IDs or query strings:  
    `https://api.example.com/users/123?active=true`

## **Examples**
- `GET /users` – Retrieve a list of users
- `POST /users` – Create a new user
- `GET /users/{id}` – Retrieve a specific user by ID
- `PUT /users/{id}` – Update a user by ID
- `DELETE /users/{id}` – Delete a user by ID

Endpoints, combined with HTTP methods, define how clients interact with the API's resources.


Points: 
* API keys per client 
* Throttling
* Request Validation
* Private API end points
* Rate Limiting
* End usage throttling
* Execution Monitoring
* Integrations

Rest /web/soap
- architecture, structures, use cases
* Rest
* web based api - representational state transfer
* web/mobile/desktop
* based on http ( standard protocol for communicating over internet)
* get,post,put,delete 
* follows stateless client-server model
* server doesn't store any information about the client's state between requests.
* web services/ stateless, scalable, crud operations
* real time communication, 
* streaming services

web apis
* broader term
* includes (rest,soap, xml-rps etc)
* platform independent interface
* uses different protocols ( http,https,tcp/ip)
* doesnt follow a specific structure 
* depending on requirements it changes
* flexible interface 
* crud, authentication, authorization and security
* powerful for building complex apps
* better when building complex apps with integratiosn
* 

SOAP 
* simple objectt access protocol
* exchanign structured data between different apps 
* xml as the format of sending and receiving data
* text,numbs, dates and binary data
* http,smtp and ftp 
* ability to define set of rules for exchanigng information
* support for advanced security ; digital signatures and encryption
* when : high secure and reliable method 
* enterprise level apps 
* complex data structures and buisness logic


api gateway 
proxy
reverse proxy
web hooks
osi model
load balancer 
firewall


business models


starlet framework
pydantic for data validation
