candidate_models = [
                    "gemini-3.8-flash",
                    "gemini-2.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-2.0-flash-exp"
                ]

                response_content = None
                successful_model = None
                last_error = None

                for model_name in candidate_models:
                    try:
                        llm_vision = ChatGoogleGenerativeAI(
                            google_api_key=gemini_api_key,
                            model=model_name,
                            temperature=0.1
                        )
                        res = llm_vision.invoke([HumanMessage(content=content_payload)])
                        response_content = res.content
                        successful_model = model_name
                        break  # Stop at first successful response
                    except Exception as e:
                        last_error = str(e)
                        continue

                if response_content:
                    st.divider()
                    st.subheader("💡 Answer")

                    # Handle case where response.content is a list of blocks/dicts instead of a simple string
                    parsed_text = ""
                    if isinstance(response_content, list):
                        for block in response_content:
                            if isinstance(block, dict) and "text" in block:
                                parsed_text += block["text"]
                            elif isinstance(block, str):
                                parsed_text += block
                    else:
                        parsed_text = str(response_content)

                    # Clean up escaped newlines if returned as raw string literal
                    parsed_text = parsed_text.replace("\\n", "\n")

                    # Render answer in a styled box/card UI
                    with st.container(border=True):
                        st.markdown(parsed_text)
                    
                    st.caption(f"⚡ *Generated using `{successful_model}`*")
                else:
                    st.error(f"All candidate Vision models failed. Last error: {last_error}")
