document.addEventListener("DOMContentLoaded", () => {
    initRevealEffects();
    initToasts();
    initConfirmModal();
    initSubmitLoader();
});

function initRevealEffects() {
    const revealTargets = Array.from(document.querySelectorAll(".hero, .panel, .card, .table-wrap"));
    if (!revealTargets.length) {
        return;
    }

    revealTargets.forEach((element, index) => {
        if (!element.hasAttribute("data-reveal")) {
            element.setAttribute("data-reveal", "true");
        }
        if (!element.style.getPropertyValue("--reveal-delay")) {
            const delayMs = Math.min(index * 45, 260);
            element.style.setProperty("--reveal-delay", `${delayMs}ms`);
        }
    });

    if (!("IntersectionObserver" in window)) {
        revealTargets.forEach((element) => element.classList.add("is-visible"));
        return;
    }

    const observer = new IntersectionObserver(
        (entries, obs) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    obs.unobserve(entry.target);
                }
            });
        },
        {
            threshold: 0.2,
            rootMargin: "0px 0px -30px 0px",
        }
    );

    revealTargets.forEach((element) => observer.observe(element));
}

function initToasts() {
    const toasts = Array.from(document.querySelectorAll("[data-toast]"));
    if (!toasts.length) {
        return;
    }

    const dismissToast = (toastElement) => {
        if (!toastElement || toastElement.dataset.dismissing === "true") {
            return;
        }
        toastElement.dataset.dismissing = "true";
        toastElement.classList.add("is-leaving");
        window.setTimeout(() => {
            toastElement.remove();
        }, 240);
    };

    document.addEventListener("click", (event) => {
        if (!(event.target instanceof Element)) {
            return;
        }

        const closeBtn = event.target.closest("[data-dismiss-toast]");
        if (!closeBtn) {
            return;
        }

        dismissToast(closeBtn.closest("[data-toast]"));
    });

    toasts.forEach((toastElement, index) => {
        window.setTimeout(() => dismissToast(toastElement), 4600 + index * 650);
    });
}

function addSubmitterMirror(form, submitter) {
    if (!submitter || !submitter.name) {
        return;
    }

    const previous = form.querySelector("input[data-submitter-mirror='true']");
    if (previous) {
        previous.remove();
    }

    const mirror = document.createElement("input");
    mirror.type = "hidden";
    mirror.name = submitter.name;
    mirror.value = submitter.value;
    mirror.dataset.submitterMirror = "true";
    form.appendChild(mirror);
}

function initConfirmModal() {
    const modal = document.getElementById("confirm-modal");
    const backdrop = document.getElementById("confirm-backdrop");
    if (!modal || !backdrop) {
        return;
    }

    const messageNode = document.getElementById("confirm-message");
    const confirmBtn = modal.querySelector("[data-modal-confirm]");
    const cancelBtn = modal.querySelector("[data-modal-cancel]");

    if (!confirmBtn || !cancelBtn || !messageNode) {
        return;
    }

    let pendingButton = null;
    let hideTimer = null;

    const openModal = (message) => {
        if (hideTimer) {
            window.clearTimeout(hideTimer);
            hideTimer = null;
        }

        messageNode.textContent = message || "Are you sure you want to continue?";
        modal.hidden = false;
        backdrop.hidden = false;
        modal.classList.add("is-visible");
        backdrop.classList.add("is-visible");
        document.body.classList.add("modal-open");
    };

    const closeModal = () => {
        modal.classList.remove("is-visible");
        backdrop.classList.remove("is-visible");
        document.body.classList.remove("modal-open");

        hideTimer = window.setTimeout(() => {
            modal.hidden = true;
            backdrop.hidden = true;
            hideTimer = null;
        }, 200);
    };

    const submitWithButton = (buttonElement) => {
        const form = buttonElement.form;
        if (!form) {
            return;
        }

        addSubmitterMirror(form, buttonElement);

        if (typeof form.requestSubmit === "function") {
            form.requestSubmit(buttonElement);
            return;
        }

        form.submit();
    };

    document.addEventListener("click", (event) => {
        if (!(event.target instanceof Element)) {
            return;
        }

        const confirmTrigger = event.target.closest("[data-confirm-submit]");
        if (!confirmTrigger) {
            return;
        }

        if (!(confirmTrigger instanceof HTMLButtonElement || confirmTrigger instanceof HTMLInputElement)) {
            return;
        }

        event.preventDefault();
        pendingButton = confirmTrigger;
        openModal(confirmTrigger.getAttribute("data-confirm-submit") || "Are you sure you want to continue?");
    });

    confirmBtn.addEventListener("click", () => {
        if (!pendingButton) {
            closeModal();
            return;
        }

        const buttonToSubmit = pendingButton;
        pendingButton = null;
        closeModal();
        submitWithButton(buttonToSubmit);
    });

    const cancelModal = () => {
        pendingButton = null;
        closeModal();
    };

    cancelBtn.addEventListener("click", cancelModal);
    backdrop.addEventListener("click", cancelModal);

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && modal.classList.contains("is-visible")) {
            cancelModal();
        }
    });
}

function initSubmitLoader() {
    document.addEventListener("submit", (event) => {
        if (!(event.target instanceof HTMLFormElement)) {
            return;
        }

        const form = event.target;
        const submitter = event.submitter;

        if (submitter instanceof HTMLButtonElement || submitter instanceof HTMLInputElement) {
            addSubmitterMirror(form, submitter);
        }

        document.body.classList.add("is-loading");

        window.setTimeout(() => {
            const submitControls = form.querySelectorAll("button[type='submit'], input[type='submit']");
            submitControls.forEach((control) => {
                if (control instanceof HTMLButtonElement || control instanceof HTMLInputElement) {
                    control.disabled = true;
                }
            });
        }, 0);
    });

    window.addEventListener("pageshow", () => {
        document.body.classList.remove("is-loading");
    });
}
