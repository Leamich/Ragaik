<!-- +page.svelte – updated chat page integrating the fresh Tailwind/Svelte UI while preserving existing API logic -->
<script lang="ts">
    import MarkdownWithMath from '$lib/MarkdownWithMath.svelte';
    import {askQuery, deleteHistory, loadHistory} from '../../api';
    import type {MessageResponseSchema, QuerySchema} from '../../types/schema';
    import {onMount} from 'svelte';

    interface Message {
        id: number;
        user: 'user' | 'bot';
        text: string;
        images?: string[];
    }

    // Initial welcome message from the bot (preserved from original page)
    let messages: Message[] = [
        {
            id: 1,
            user: 'bot',
            text: 'Hi! I\'m your AI assistant. Ask me anything!'
        }
    ];

    let input = '';
    let loading = false;
    let nextId = messages.length + 1;
    let chatContainer: HTMLDivElement;

    /** Scroll chat view to the bottom */
    function scrollToBottom() {
        requestAnimationFrame(() => {
            if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
        });
    }

    /** Restore previous chat (original logic kept) */
    onMount(async () => {
        const loadedMessages = await loadHistory();
        if (loadedMessages && loadedMessages.length > 0) {
            const restored: Message[] = loadedMessages.map((msg, idx) => ({
                id: nextId + idx,
                user: idx % 2 === 0 ? 'user' : 'bot',
                text: msg
            }));
            messages = [...messages, ...restored];
            nextId = messages.length + 1;
            scrollToBottom();
        }
    });


    /** Handle submit from textarea */
    async function handleSubmit() {
        const question = input.trim();
        if (!question || loading) return;

        // Add user message
        messages = [...messages, {id: nextId++, user: 'user', text: question}];
        input = '';
        loading = true;
        scrollToBottom();

        try {
            const payload: QuerySchema = {query: question};
            const res: MessageResponseSchema = await askQuery(payload);

            messages = [
                ...messages,
                {
                    id: nextId++,
                    user: 'bot',
                    text: res.text,
                    images: res.image_ids
                }
            ];

        } catch {
            messages = [
                ...messages,
                {
                    id: nextId++,
                    user: 'bot',
                    text: 'Sorry – something went wrong while contacting the server.'
                }
            ];
        } finally {
            loading = false;
            scrollToBottom();
        }
    }

    /** Submit on Enter (Shift+Enter for newline) */
    function onKeyDown(event: KeyboardEvent) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSubmit();
        }
    }

    /** Erase session button */
    async function handleEraseSession() {
        await deleteHistory();
        window.location.reload();
    }
</script>

<!-- MAIN LAYOUT -->
<div class="flex flex-col h-[calc(100vh-80px)] max-h-[calc(100vh-80px)] bg-white overflow-hidden py-4">
    <!-- Chat area -->
    <div
            bind:this={chatContainer}
            class="flex-1 overflow-y-auto p-4 space-y-4 bg-white chat-container"
    >
        {#each messages as msg (msg.id)}
            <div class="flex {msg.user === 'user' ? 'justify-end' : 'justify-start'}">
                <div
                        class="px-3 py-2 rounded-lg max-w-[80%] break-words whitespace-pre-wrap"
                        class:bg-violet-400={msg.user === 'user'}
                        class:text-white={msg.user === 'user'}
                        class:rounded-br-none={msg.user === 'user'}
                        class:bg-gray-200={msg.user === 'bot'}
                        class:text-gray-900={msg.user === 'bot'}
                        class:rounded-bl-none={msg.user === 'bot'}
                >
                    <!-- Render markdown+math content -->
                    <MarkdownWithMath text={msg.text}/>

                    <!-- Render assistant images if any -->
                    {#if msg.images}
                        <div class="mt-2 flex flex-col gap-2">
                            {#each msg.images as imgSrc (imgSrc)}
                                <button
                                        class="max-h-72 rounded-lg object-contain border flex justify-center items-center"
                                        on:click={() => window.open(`/api/v1/photos/${imgSrc}.png`, '_blank')}>
                                    <img
                                            src={`/api/v1/photos/${imgSrc}.png`}
                                            class="max-h-72 rounded-lg object-contain"
                                            loading="lazy"
                                            alt="Context"
                                    />
                                </button>

                            {/each}
                        </div>
                    {/if}
                </div>
            </div>
        {/each}

        <!-- Loading spinner inline with messages (optional) -->
        {#if loading}
            <div class="flex justify-start">
                <div class="w-6 h-6 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
            </div>
        {/if}
    </div>

    <!-- Input area pinned at bottom -->
    <div class="border-t bg-white p-2">
        <form class="flex items-end gap-2" on:submit|preventDefault={handleSubmit}>
      <textarea
              bind:value={input}
              class="flex-1 resize-none p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
              on:keydown={onKeyDown}
              placeholder="Type your message..."
              rows="1"
              style="min-height: 44px; max-height: 120px;"
      ></textarea>

            <!-- Send button -->
            <button
                    aria-label="Send"
                    class="relative bg-blue-500 text-white p-2 rounded disabled:opacity-50"
                    disabled={loading || !input.trim()}
                    type="submit"
            >
                {#if loading}
                    <svg
                            class="animate-spin h-5 w-5"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                    >
                        <circle
                                class="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                stroke-width="4"
                        ></circle>
                        <path
                                class="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                        ></path>
                    </svg>
                {:else}
                    <svg
                            class="h-5 w-5"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="currentColor"
                            viewBox="0 0 16 16"
                    >
                        <path
                                d="M15.854.146a.5.5 0 0 0-.54-.11l-15 6a.5.5 0 0 0 0 .928l15 6a.5.5 0 0 0 .686-.475V.621a.5.5 0 0 0-.146-.475zM6 10.117V5.883L11.114 8 6 10.117z"
                        />
                    </svg>
                {/if}
            </button>

            <!-- Quick erase button (mobile convenience) -->
            <button
                    class="bg-red-500 text-white px-2 py-2 rounded"
                    on:click={handleEraseSession}
                    type="button"
            >
                Erase
            </button>
        </form>
    </div>
</div>

<style>
    /* Thin, subtle scroll bar styling (kept from original) */
    .chat-container {
        scrollbar-width: thin;
    }

    .chat-container::-webkit-scrollbar {
        width: 0.5rem;
    }

    .chat-container::-webkit-scrollbar-thumb {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 0.25rem;
    }
</style>
