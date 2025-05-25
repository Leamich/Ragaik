<script lang="ts">
	// Sample messages for design purposes
	let messages = [
		{ id: 1, user: 'bot', text: 'Hello! How can I assist you today?' },
		{ id: 2, user: 'user', text: 'I want to learn more about SvelteKit.' },
		{ id: 3, user: 'bot', text: 'Sure! SvelteKit is a powerful framework ...' }
	];
</script>

<div class="relative h-screen bg-gray-50">
	<!-- Chat messages with typography styling -->
	<div class="prose prose-slate max-w-full px-4 py-6 overflow-y-auto h-full chat-container pb-32">
		{#each messages as msg (msg.id)}
			<div
				id="msg-{msg.id}"
				class="mb-4 p-3 rounded-lg shadow-sm"
				class:bg-gray-100={msg.user === 'bot'}
				class:bg-blue-50={msg.user === 'user'}
			>
				<!-- eslint-disable-next-line svelte/no-at-html-tags -->
				<p>{@html msg.text}</p>
			</div>
		{/each}
	</div>

	<!-- Prompt overlay -->
	<div class="absolute bottom-0 left-0 right-0 bg-white bg-opacity-90 backdrop-blur-md p-4">
		<form class="flex items-center space-x-2">
			<textarea
				rows="1"
				placeholder="Type your message..."
				class="flex-1 p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring focus:ring-indigo-200 resize-none"
			></textarea>
			<button
				type="submit"
				class="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring focus:ring-indigo-200"
			>
				Send
			</button>
		</form>
	</div>
</div>

<style>
	/* Ensure the messages container scrolls beneath the overlay prompt */
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
