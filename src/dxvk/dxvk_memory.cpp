// This is the ONE and ONLY translation unit that provides the actual
// VMA function bodies. VMA_DXVK_IMPLEMENTATION unlocks VMA_IMPLEMENTATION
// inside dxvk_memory.h's include guard (see that file for why this is
// needed) even if some global build flag also defines VMA_IMPLEMENTATION
// elsewhere; both macros are undone immediately after the include so
// nothing downstream in this file is affected.
#define VMA_DXVK_IMPLEMENTATION
#define VMA_IMPLEMENTATION

#include <algorithm>
#include <array>

#include "dxvk_device.h"
#include "dxvk_memory.h"

#undef VMA_IMPLEMENTATION
#undef VMA_DXVK_IMPLEMENTATION

namespace dxvk {

  namespace {
    // Combines VMA's own per-heap budget statistics with bytes tracked
    // separately for allocations that bypass VMA (see DxvkMemory::isRaw).
    // Factored out so both getMemoryStats() and the allocation-failure
    // diagnostics in DxvkMemoryAllocator::alloc() compute this the same
    // way from a single vmaGetHeapBudgets() call, rather than each heap
    // re-querying VMA's (non-trivial) budget computation on its own.
    DxvkMemoryStats computeMemoryStats(const VmaBudget& budget, VkDeviceSize rawBytes) {
      DxvkMemoryStats stats;
      stats.memoryAllocated = budget.statistics.blockBytes      + rawBytes;
      stats.memoryUsed      = budget.statistics.allocationBytes + rawBytes;
      return stats;
    }

    // Maps GPU priority hints onto VMA/VK_EXT_memory_priority's [0, 1]
    // priority range. Shared between the VMA-routed path (tryAlloc) and
    // the raw vkAllocateMemory path (tryAllocRaw) so the two can't drift
    // apart from each other.
    float priorityFromHints(DxvkMemoryFlags hints) {
      if (hints.test(DxvkMemoryFlag::GpuWritable))
        return 1.0f;
      if (hints.test(DxvkMemoryFlag::GpuReadable))
        return 0.5f;
      return 0.0f;
    }
  }


  DxvkMemory::DxvkMemory() { }
  DxvkMemory::DxvkMemory(
          DxvkMemoryAllocator*  alloc,
          VmaAllocation         allocation,
          VkDeviceMemory        memory,
          VkDeviceSize          offset,
          VkDeviceSize          length,
          void*                 mapPtr,
          bool                  isRaw,
          uint32_t              memoryTypeIndex)
  : m_alloc           (alloc),
    m_allocation      (allocation),
    m_memory          (memory),
    m_offset          (offset),
    m_length          (length),
    m_mapPtr          (mapPtr),
    m_isRaw           (isRaw),
    m_memoryTypeIndex (memoryTypeIndex) { }


  DxvkMemory::DxvkMemory(DxvkMemory&& other)
  : m_alloc           (std::exchange(other.m_alloc,      nullptr)),
    m_allocation      (std::exchange(other.m_allocation, VmaAllocation(VK_NULL_HANDLE))),
    m_memory          (std::exchange(other.m_memory,     VkDeviceMemory(VK_NULL_HANDLE))),
    m_offset          (std::exchange(other.m_offset,     0)),
    m_length          (std::exchange(other.m_length,     0)),
    m_mapPtr          (std::exchange(other.m_mapPtr,     nullptr)),
    m_isRaw           (std::exchange(other.m_isRaw,      false)),
    m_memoryTypeIndex (std::exchange(other.m_memoryTypeIndex, 0)) { }


  DxvkMemory& DxvkMemory::operator = (DxvkMemory&& other) {
    this->free();
    m_alloc           = std::exchange(other.m_alloc,      nullptr);
    m_allocation      = std::exchange(other.m_allocation, VmaAllocation(VK_NULL_HANDLE));
    m_memory          = std::exchange(other.m_memory,     VkDeviceMemory(VK_NULL_HANDLE));
    m_offset          = std::exchange(other.m_offset,     0);
    m_length          = std::exchange(other.m_length,     0);
    m_mapPtr          = std::exchange(other.m_mapPtr,     nullptr);
    m_isRaw           = std::exchange(other.m_isRaw,      false);
    m_memoryTypeIndex = std::exchange(other.m_memoryTypeIndex, 0);
    return *this;
  }


  DxvkMemory::~DxvkMemory() {
    this->free();
  }


  void DxvkMemory::free() {
    if (m_memory == VK_NULL_HANDLE)
      return;

    if (m_isRaw) {
      // Bypassed VMA at allocation time (shared/external memory), so we
      // have to bypass it here too, and also mirror the free back to the
      // adapter/heap accounting ourselves since VMA's device memory
      // callbacks never see this allocation either.
      m_alloc->notifyRawFree(m_memoryTypeIndex, m_length);
      m_alloc->m_vkd->vkFreeMemory(m_alloc->m_vkd->device(), m_memory, nullptr);
    } else {
      vmaFreeMemory(m_alloc->handle(), m_allocation);
    }
  }


  DxvkMemoryAllocator::DxvkMemoryAllocator(const DxvkDevice* device)
  : m_vkd             (device->vkd()),
    m_device          (device),
    m_devProps        (device->adapter()->deviceProperties()),
    m_memProps        (device->adapter()->memoryProperties()),
    m_isUMA           (device->isUnifiedMemoryArchitecture()) {
    // See the m_rawAllocated declaration in the header for why this is
    // zeroed explicitly instead of via a default member initializer.
    for (auto& counter : m_rawAllocated)
      counter = 0;

    // NOTE: adjust the accessor names below (vki(), instance()->handle(),
    // adapter()->handle(), device->handle()) if they differ from the
    // ones used in your copy of dxvk_adapter.h / dxvk_device.h.
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr                    = nullptr;
    vulkanFunctions.vkGetDeviceProcAddr                      = nullptr;
    vulkanFunctions.vkGetPhysicalDeviceProperties            = device->adapter()->vki()->vkGetPhysicalDeviceProperties;
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties      = device->adapter()->vki()->vkGetPhysicalDeviceMemoryProperties;
    vulkanFunctions.vkAllocateMemory                         = m_vkd->vkAllocateMemory;
    vulkanFunctions.vkFreeMemory                             = m_vkd->vkFreeMemory;
    vulkanFunctions.vkMapMemory                              = m_vkd->vkMapMemory;
    vulkanFunctions.vkUnmapMemory                            = m_vkd->vkUnmapMemory;
    vulkanFunctions.vkFlushMappedMemoryRanges                = m_vkd->vkFlushMappedMemoryRanges;
    vulkanFunctions.vkInvalidateMappedMemoryRanges           = m_vkd->vkInvalidateMappedMemoryRanges;
    vulkanFunctions.vkBindBufferMemory                       = m_vkd->vkBindBufferMemory;
    vulkanFunctions.vkBindImageMemory                        = m_vkd->vkBindImageMemory;
    vulkanFunctions.vkGetBufferMemoryRequirements            = m_vkd->vkGetBufferMemoryRequirements;
    vulkanFunctions.vkGetImageMemoryRequirements             = m_vkd->vkGetImageMemoryRequirements;
    vulkanFunctions.vkCreateBuffer                           = m_vkd->vkCreateBuffer;
    vulkanFunctions.vkDestroyBuffer                          = m_vkd->vkDestroyBuffer;
    vulkanFunctions.vkCreateImage                            = m_vkd->vkCreateImage;
    vulkanFunctions.vkDestroyImage                           = m_vkd->vkDestroyImage;
    vulkanFunctions.vkCmdCopyBuffer                          = m_vkd->vkCmdCopyBuffer;
    vulkanFunctions.vkGetBufferMemoryRequirements2KHR        = m_vkd->vkGetBufferMemoryRequirements2;
    vulkanFunctions.vkGetImageMemoryRequirements2KHR         = m_vkd->vkGetImageMemoryRequirements2;
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR  = device->adapter()->vki()->vkGetPhysicalDeviceMemoryProperties2;

    // vk::DeviceFn doesn't declare vkBindBufferMemory2 / vkBindImageMemory2
    // as named members, but it (via its DeviceLoader base) exposes a
    // generic sym(name) getter that goes through the very same per-device
    // vkGetDeviceProcAddr it uses internally for every other function.
    // Both are core Vulkan 1.1 functions, so this is guaranteed to succeed
    // since 1.1 is this allocator's minimum supported version.
    using PFN_vmaBindBufferMemory2 = VkResult (VKAPI_PTR *)(VkDevice, uint32_t, const VkBindBufferMemoryInfo*);
    using PFN_vmaBindImageMemory2  = VkResult (VKAPI_PTR *)(VkDevice, uint32_t, const VkBindImageMemoryInfo*);

    auto pfnBindBufferMemory2 = reinterpret_cast<PFN_vmaBindBufferMemory2>(m_vkd->sym("vkBindBufferMemory2"));
    auto pfnBindImageMemory2  = reinterpret_cast<PFN_vmaBindImageMemory2>(m_vkd->sym("vkBindImageMemory2"));

    vulkanFunctions.vkBindBufferMemory2KHR = pfnBindBufferMemory2;
    vulkanFunctions.vkBindImageMemory2KHR  = pfnBindImageMemory2;

    // Since we target Vulkan 1.1+ (see vulkanApiVersion below), dedicated
    // allocation and vkGetPhysicalDeviceMemoryProperties2 are core and are
    // picked up automatically by VMA from vulkanApiVersion; no explicit
    // VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT is needed.
    VmaAllocatorCreateFlags allocatorFlags =
        VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

    // Only ask VMA to use vkBind*Memory2 if both entry points actually
    // resolved; falls back to the always-available non-2 bind calls
    // otherwise instead of risking a null function pointer call.
    if (pfnBindBufferMemory2 && pfnBindImageMemory2)
      allocatorFlags |= VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;

    if (m_device->features().extMemoryPriority.memoryPriority)
      allocatorFlags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;

    // VK_KHR_buffer_device_address (core in 1.2) is already tracked as a
    // device feature for CUDA interop; if it's enabled, tell VMA so it
    // sets VK_MEMORY_ALLOCATE_FLAGS_INFO::deviceAddress on every relevant
    // allocation automatically instead of leaving that to callers.
    if (m_device->features().khrBufferDeviceAddress.bufferDeviceAddress)
      allocatorFlags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    // Per-heap size limits replace the manual 'budget' bookkeeping
    // that the old free-list allocator did by hand.
    std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heapSizeLimits;
    heapSizeLimits.fill(VK_WHOLE_SIZE);

    // Tell VMA which memory types can be exported/imported as Win32
    // handles. DxvkSharedHandleInfo (see dxvk_memory.h) already models
    // Import/Export for D3D11-style shared resources, and vk::DeviceFn
    // already has vkGetMemoryWin32HandleKHR wired up — this is the piece
    // that lets VMA correctly chain VkExportMemoryAllocateInfoKHR for
    // allocations that need it, instead of only being usable for purely
    // internal (non-shared) memory. Declared here so it stays alive for
    // the vmaCreateAllocator() call below; only actually populated if the
    // device extension is enabled.
    std::array<VkExternalMemoryHandleTypeFlags, VK_MAX_MEMORY_TYPES> externalMemoryHandleTypes = { };

    if (m_device->extensions().khrExternalMemoryWin32) {
      for (uint32_t i = 0; i < m_memProps.memoryTypeCount; i++)
        externalMemoryHandleTypes[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    }

    /* Target 80% of a heap on UMA systems where system RAM and VRAM are
     * the same pool and we must leave room for the OS/other apps.
     *
     * On discrete GPUs, still leave a small 5% safety margin below the
     * heap's nominal size. Letting allocations creep up to literally
     * 100% of VRAM invites the driver to start silently paging /
     * compressing / evicting memory to make room, which shows up as
     * exactly the kind of random frame hitches and delayed texture
     * uploads this is meant to avoid — better to fail an allocation
     * cleanly (and fall back to a different memory type, see alloc())
     * than to let the driver deal with an overcommitted heap. */
    for (uint32_t i = 0; i < m_memProps.memoryHeapCount; i++) {
      if (m_memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        if (m_isUMA) {
          // On UMA this heap IS system RAM: shared with the OS, the CPU
          // side of this app, and every other process running. Without
          // VK_EXT_memory_budget we have no visibility into how much of
          // it other processes are already using, so leave extra
          // headroom in that case rather than risk system-wide paging/
          // thrashing instead of just a clean allocation failure here.
          heapSizeLimits[i] = m_device->extensions().extMemoryBudget
            ? (8 * m_memProps.memoryHeaps[i].size) / 10   // UMA + budget info:   80%
            : (7 * m_memProps.memoryHeaps[i].size) / 10;  // UMA, no budget info: 70%
        } else {
          heapSizeLimits[i] = (95 * m_memProps.memoryHeaps[i].size) / 100; // discrete: 95%
        }
      }
    }

    /* Check what kind of heap the HVV memory type is on, if any. If the
     * HVV memory type is on the largest device-local heap, we either have
     * an UMA system or an RBAR-enabled system. Otherwise, there will likely
     * be a separate, smaller heap for it. */
    VkDeviceSize largestDeviceLocalHeap = 0;

    for (uint32_t i = 0; i < m_memProps.memoryTypeCount; i++) {
      if (m_memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        largestDeviceLocalHeap = std::max(largestDeviceLocalHeap,
          m_memProps.memoryHeaps[m_memProps.memoryTypes[i].heapIndex].size);
    }

    /* Work around an issue on Nvidia drivers where using the entire
     * device_local | host_visible heap can cause crashes or slowdowns */
    if (m_device->properties().core.properties.vendorID == uint16_t(DxvkGpuVendor::Nvidia)) {
      bool shrinkNvidiaHvvHeap = device->adapter()->matchesDriver(DxvkGpuVendor::Nvidia,
        VK_DRIVER_ID_NVIDIA_PROPRIETARY_KHR, 0, VK_MAKE_VERSION(465, 0, 0));

      applyTristate(shrinkNvidiaHvvHeap, device->config().shrinkNvidiaHvvHeap);

      if (shrinkNvidiaHvvHeap) {
        VkMemoryPropertyFlags hvvFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        for (uint32_t i = 0; i < m_memProps.memoryTypeCount; i++) {
          uint32_t heapIndex = m_memProps.memoryTypes[i].heapIndex;

          if ((m_memProps.memoryTypes[i].propertyFlags & hvvFlags) == hvvFlags
           && (m_memProps.memoryHeaps[heapIndex].size < largestDeviceLocalHeap))
            heapSizeLimits[heapIndex] = std::min(heapSizeLimits[heapIndex], VkDeviceSize(32 << 20));
        }
      }
    }

    // DxvkAdapter doesn't keep a back-reference to the DxvkInstance, but
    // vk::InstanceFn (like vk::DeviceFn::device()) already carries the
    // VkInstance handle it was created from.
    VkInstance vkInstance = device->adapter()->vki()->instance();

    // Use Vulkan 1.2 if both the instance and the physical device report
    // support for it, so VMA can rely on core (rather than *_KHR) entry
    // points where it matters; otherwise fall back to 1.1, which is our
    // minimum supported version and already gives VMA dedicated allocation
    // and vkGetXMemoryRequirements2/vkGetPhysicalDeviceMemoryProperties2
    // as core functionality.
    uint32_t vmaApiVersion = VK_API_VERSION_1_1;

    if (m_devProps.apiVersion >= VK_API_VERSION_1_2)
      vmaApiVersion = VK_API_VERSION_1_2;

    // Mirrors every real vkAllocateMemory/vkFreeMemory call VMA makes
    // (i.e. once per underlying block, not once per suballocation) back
    // to DxvkAdapter's own heap-usage counters. This is what keeps
    // DxvkAdapter::getMemoryHeapInfo() (used by the failure diagnostics
    // below, and by anything querying DXGI-style budget info) accurate
    // on devices that don't support VK_EXT_memory_budget -- without it,
    // DxvkAdapter::m_heapAlloc never gets incremented and always reads 0.
    VmaDeviceMemoryCallbacks deviceMemoryCallbacks = {};
    deviceMemoryCallbacks.pfnAllocate = &DxvkMemoryAllocator::deviceMemoryAllocateCallback;
    deviceMemoryCallbacks.pfnFree     = &DxvkMemoryAllocator::deviceMemoryFreeCallback;
    deviceMemoryCallbacks.pUserData   = this;

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.flags                  = allocatorFlags;
    allocatorInfo.physicalDevice         = device->adapter()->handle();
    allocatorInfo.device                 = device->handle();
    allocatorInfo.instance               = vkInstance;
    allocatorInfo.vulkanApiVersion       = vmaApiVersion;
    allocatorInfo.pVulkanFunctions       = &vulkanFunctions;
    allocatorInfo.pHeapSizeLimit         = heapSizeLimits.data();
    allocatorInfo.pDeviceMemoryCallbacks = &deviceMemoryCallbacks;

    if (m_device->extensions().khrExternalMemoryWin32)
      allocatorInfo.pTypeExternalMemoryHandleTypes = externalMemoryHandleTypes.data();

    // Mirrors roughly what the old free-list allocator used, but smaller.
    // Every time VMA needs a brand new block it has to call vkAllocateMemory
    // synchronously, and that single driver call can itself take several
    // milliseconds — a big contributor to the kind of sudden one-frame
    // hitches you get right when a new texture/buffer needs fresh VRAM.
    // A smaller block means each individual stall is shorter, at the cost
    // of needing somewhat more of them; for a game that's already stalling
    // sometimes, more-but-shorter beats fewer-but-longer.
    allocatorInfo.preferredLargeHeapBlockSize = 64ull << 20;

    if (vmaCreateAllocator(&allocatorInfo, &m_vma) != VK_SUCCESS)
      throw DxvkError("DxvkMemoryAllocator: Failed to create VMA allocator");
  }


  DxvkMemoryAllocator::~DxvkMemoryAllocator() {
    for (auto& poolsOfCategory : m_pools) {
      for (auto& [memTypeIndex, pool] : poolsOfCategory)
        vmaDestroyPool(m_vma, pool);
    }

    if (m_vma != VK_NULL_HANDLE)
      vmaDestroyAllocator(m_vma);
  }


  DxvkMemory DxvkMemoryAllocator::alloc(
    const VkMemoryRequirements*             req,
    const VkMemoryDedicatedRequirements&    dedAllocReq,
    const VkMemoryDedicatedAllocateInfo&    dedAllocInfo,
          VkMemoryPropertyFlags             flags,
          DxvkMemoryFlags                   hints) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);

    // Keep small allocations together to avoid fragmenting
    // chunks for larger resources with lots of small gaps,
    // as well as resources with potentially weird lifetimes.
    // VMA still benefits from this hint since it influences
    // which internal block list the allocation is placed in.
    if (req->size <= SmallAllocationThreshold) {
      hints.set(DxvkMemoryFlag::Small);
      hints.clr(DxvkMemoryFlag::GpuWritable, DxvkMemoryFlag::GpuReadable);
    }

    // Ignore most hints for host-visible allocations since they usually
    // don't make much sense for those resources -- except on a UMA
    // system, where host-visible memory is commonly also the fast,
    // GPU-resident memory (there's often no separate, lower-priority
    // "staging" pool for it to fall back to). Stripping the GPU-priority
    // hints there would tell VMA's pooling/priority/strategy logic to
    // treat what may be a hot, frequently-GPU-accessed resource the same
    // as plain staging memory, so keep the hints intact in that case.
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && !m_isUMA)
      hints = hints & DxvkMemoryFlag::Transient;

    // Try to allocate with the exact requested property flags first
    DxvkMemory result = this->tryAlloc(req, dedAllocReq, dedAllocInfo, flags, hints);

    // If the first attempt failed and the dedicated allocation was only
    // preferred (not required), retry while allowing VMA to place it
    // inside a regular block instead.
    if (!result && dedAllocReq.prefersDedicatedAllocation && !dedAllocReq.requiresDedicatedAllocation) {
      VkMemoryDedicatedRequirements relaxedReq = dedAllocReq;
      relaxedReq.prefersDedicatedAllocation = VK_FALSE;

      VkMemoryDedicatedAllocateInfo emptyDedInfo = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
      result = this->tryAlloc(req, relaxedReq, emptyDedInfo, flags, hints);
    }

    // If that still didn't work, probe increasingly relaxed sets of
    // optional property flags. Order matters: give up on the least
    // impactful property first. Losing HOST_CACHED_BIT only affects
    // host access performance; losing DEVICE_LOCAL_BIT moves the
    // allocation off the GPU entirely, so that has to be the very
    // last thing this ladder gives up on.
    static constexpr VkMemoryPropertyFlags relaxationOrder[] = {
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };

    VkMemoryPropertyFlags remFlags = 0;

    for (VkMemoryPropertyFlags bit : relaxationOrder) {
      if (result || !(flags & bit))
        continue;

      remFlags |= bit;
      result = this->tryAlloc(req, dedAllocReq, dedAllocInfo, flags & ~remFlags, hints);
    }

    if (!result) {
      DxvkAdapterMemoryInfo memHeapInfo = m_device->adapter()->getMemoryHeapInfo();

      std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> vmaBudgets = { };
      vmaGetHeapBudgets(m_vma, vmaBudgets.data());

      Logger::err(str::format(
        "DxvkMemoryAllocator: Memory allocation failed",
        "\n  Size:      ", req->size,
        "\n  Alignment: ", req->alignment,
        "\n  Mem flags: ", "0x", std::hex, flags,
        "\n  Mem types: ", "0x", std::hex, req->memoryTypeBits));

      for (uint32_t i = 0; i < m_memProps.memoryHeapCount; i++) {
        DxvkMemoryStats stats = computeMemoryStats(vmaBudgets[i],
          m_rawAllocated[i].load(std::memory_order_relaxed));

        Logger::err(str::format("Heap ", i, ": ",
          (stats.memoryAllocated >> 20), " MB allocated, ",
          (stats.memoryUsed      >> 20), " MB used, ",
          m_device->extensions().extMemoryBudget
            ? str::format(
                (memHeapInfo.heaps[i].memoryAllocated >> 20), " MB allocated (driver), ",
                (memHeapInfo.heaps[i].memoryBudget    >> 20), " MB budget (driver), ",
                (m_memProps.memoryHeaps[i].size        >> 20), " MB total")
            : str::format(
                (m_memProps.memoryHeaps[i].size        >> 20), " MB total")));
      }

      // Dump VMA's full internal state (every block, every pool, current
      // fragmentation) to the log. Relatively expensive, but we're
      // already on the "allocation permanently failed, about to throw"
      // path, so the extra cost is a non-issue and the detail is exactly
      // what's needed to tell a fragmented-but-technically-free heap
      // apart from a genuinely exhausted one.
      char* statsString = nullptr;
      vmaBuildStatsString(m_vma, &statsString, VK_TRUE);

      if (statsString != nullptr) {
        Logger::err("DxvkMemoryAllocator: Detailed VMA state follows:");
        Logger::err(statsString);
        vmaFreeStatsString(m_vma, statsString);
      }

      throw DxvkError("DxvkMemoryAllocator: Memory allocation failed");
    }

    return result;
  }


  DxvkMemory DxvkMemoryAllocator::tryAlloc(
    const VkMemoryRequirements*             req,
    const VkMemoryDedicatedRequirements&    dedAllocReq,
    const VkMemoryDedicatedAllocateInfo&    dedAllocInfo,
          VkMemoryPropertyFlags             flags,
          DxvkMemoryFlags                   hints) {
    // Shared/external memory (DxvkSharedHandleMode::Import or Export,
    // see dxvk_image.cpp) needs a pNext chain — VkExportMemoryAllocateInfo
    // or VkImportMemoryWin32HandleInfoKHR — hung directly off dedAllocInfo.
    // VMA's public allocation API has no parameter for an arbitrary extra
    // pNext struct, so if we routed this through VMA that chain would
    // silently get dropped and the resulting handle would be useless for
    // interop. Detect that case up front and bypass VMA entirely for it.
    if (dedAllocInfo.pNext != nullptr)
      return this->tryAllocRaw(req, dedAllocInfo, flags, hints);

    VmaAllocationCreateInfo createInfo = {};
    createInfo.requiredFlags = flags;
    createInfo.usage         = VMA_MEMORY_USAGE_UNKNOWN;

    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      createInfo.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;

    if (dedAllocReq.requiresDedicatedAllocation || dedAllocReq.prefersDedicatedAllocation)
      createInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    // If we're still asking for VRAM (device-local) at this point, ask
    // VMA to fail cleanly if this would push us past the live memory
    // budget (VK_EXT_memory_budget, or VMA's own heap-size estimate if
    // that extension isn't present) instead of letting it succeed into
    // an overcommitted heap. The alloc() retry ladder already knows how
    // to fall back to a different, less-optimal memory type when
    // tryAlloc() fails — so this turns "driver silently starts paging
    // things around later, at a totally unpredictable time" into "we
    // notice right now and deliberately pick host memory instead".
    // Skipped when a dedicated allocation is *required* (not just
    // preferred), since that can't fall back to anything else anyway.
    if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !dedAllocReq.requiresDedicatedAllocation)
      createInfo.flags |= VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

    // Map GPU priority hints onto VMA's allocation priority.
    // Only takes effect if VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT
    // was enabled when the allocator was created.
    createInfo.priority = priorityFromHints(hints);

    // Pick a placement strategy that matches the resource's lifetime:
    //  - Small / Transient resources churn a lot (created and destroyed
    //    constantly), so we favour allocation speed over tight packing.
    //  - Long-lived GPU-priority resources are the opposite: they stick
    //    around for the lifetime of the app, so it's worth spending a
    //    bit more time packing them tightly to reduce VRAM fragmentation.
    // Resources that hit neither case are left on VMA's default heuristic.
    if (hints.test(DxvkMemoryFlag::Small) || hints.test(DxvkMemoryFlag::Transient))
      createInfo.flags |= VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT;
    else if (hints.test(DxvkMemoryFlag::GpuWritable) || hints.test(DxvkMemoryFlag::GpuReadable))
      createInfo.flags |= VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;

    VmaAllocation     allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo = {};
    VkResult          status;

    // If the caller already knows this maps to a specific image or
    // buffer, let VMA drive its own dedicated-allocation path for it.
    // Dedicated allocations always get their own VkDeviceMemory object
    // regardless of pools, so there's nothing to gain by routing them
    // through one.
    if (dedAllocInfo.image != VK_NULL_HANDLE) {
      status = vmaAllocateMemoryForImage(m_vma,
        dedAllocInfo.image, &createInfo, &allocation, &allocationInfo);
    } else if (dedAllocInfo.buffer != VK_NULL_HANDLE) {
      status = vmaAllocateMemoryForBuffer(m_vma,
        dedAllocInfo.buffer, &createInfo, &allocation, &allocationInfo);
    } else {
      // Route regular (non-dedicated) allocations through a category-
      // specific pool when we can. Each category gets its own block list
      // and its own block-size tuning (see DxvkMemoryPoolCategory), which
      // keeps e.g. a burst of small transient allocations from sharing
      // (and fragmenting) blocks with long-lived textures. This does NOT
      // change locking behaviour — we're still fully serialized by
      // m_mutex in alloc() — it only changes how the allocated space
      // itself is organized.
      uint32_t memTypeIndex = 0;

      if (vmaFindMemoryTypeIndex(m_vma, req->memoryTypeBits, &createInfo, &memTypeIndex) == VK_SUCCESS) {
        DxvkMemoryPoolCategory category = DxvkMemoryPoolCategory::Generic;

        if (hints.test(DxvkMemoryFlag::Small) || hints.test(DxvkMemoryFlag::Transient))
          category = DxvkMemoryPoolCategory::Small;
        else if (hints.test(DxvkMemoryFlag::GpuWritable) || hints.test(DxvkMemoryFlag::GpuReadable))
          category = DxvkMemoryPoolCategory::Resident;

        VmaPool pool = this->findOrCreatePool(category, memTypeIndex);

        // If pool creation ever fails, fall back to VMA's own default
        // (non-custom) pools for this memory type rather than failing
        // the whole allocation.
        if (pool != VK_NULL_HANDLE)
          createInfo.pool = pool;
      }

      status = vmaAllocateMemory(m_vma,
        req, &createInfo, &allocation, &allocationInfo);
    }

    if (status != VK_SUCCESS)
      return DxvkMemory();

    return DxvkMemory(this, allocation,
      allocationInfo.deviceMemory, allocationInfo.offset,
      allocationInfo.size, allocationInfo.pMappedData);
  }


  DxvkMemory DxvkMemoryAllocator::tryAllocRaw(
    const VkMemoryRequirements*             req,
    const VkMemoryDedicatedAllocateInfo&    dedAllocInfo,
          VkMemoryPropertyFlags             flags,
          DxvkMemoryFlags                   hints) {
    // Find a matching memory type by hand since we're bypassing VMA's
    // own selection logic for this allocation. Same exact-match rule
    // VMA itself would apply via requiredFlags.
    uint32_t memTypeIndex = UINT32_MAX;

    for (uint32_t i = 0; i < m_memProps.memoryTypeCount; i++) {
      bool supported = (req->memoryTypeBits & (1u << i)) != 0;
      bool adequate  = (m_memProps.memoryTypes[i].propertyFlags & flags) == flags;

      if (supported && adequate) {
        memTypeIndex = i;
        break;
      }
    }

    if (memTypeIndex == UINT32_MAX)
      return DxvkMemory();

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    // Chains straight into dedAllocInfo, which the caller (DxvkImage /
    // DxvkBuffer) has already chained its own VkExportMemoryAllocateInfo
    // or VkImportMemoryWin32HandleInfoKHR onto — that's the whole point
    // of this bypass path.
    allocInfo.pNext           = &dedAllocInfo;
    allocInfo.allocationSize  = req->size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    // VMA applies VK_EXT_memory_priority automatically for pooled/VMA-
    // routed allocations (see priorityFromHints() in tryAlloc() above),
    // but this path bypasses VMA entirely, so nothing sets a priority
    // here otherwise. Shared/exported images in particular (the main
    // reason this path exists at all -- see the pNext check in
    // tryAlloc()) are very often GPU-writable, performance-relevant
    // resources: render targets shared with another process, video
    // encode surfaces, and so on. Leaving them at the driver's default
    // priority would silently defeat the eviction hint for this entire
    // category of resource. Chained *before* dedAllocInfo in the pNext
    // list so the caller's own export/import chain hanging off
    // dedAllocInfo.pNext is left completely untouched.
    VkMemoryPriorityAllocateInfoEXT priorityInfo;

    if (m_device->features().extMemoryPriority.memoryPriority) {
      priorityInfo.sType    = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT;
      priorityInfo.pNext    = allocInfo.pNext;
      priorityInfo.priority = priorityFromHints(hints);
      allocInfo.pNext       = &priorityInfo;
    }

    VkDeviceMemory memory = VK_NULL_HANDLE;

    if (m_vkd->vkAllocateMemory(m_vkd->device(), &allocInfo, nullptr, &memory) != VK_SUCCESS)
      return DxvkMemory();

    // This bypasses VMA, so VMA's device memory callbacks never fire for
    // it -- account for it ourselves instead (see notifyRawAlloc).
    this->notifyRawAlloc(memTypeIndex, req->size);

    void* mapPtr = nullptr;

    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      if (m_vkd->vkMapMemory(m_vkd->device(), memory, 0, VK_WHOLE_SIZE, 0, &mapPtr) != VK_SUCCESS) {
        this->notifyRawFree(memTypeIndex, req->size);
        m_vkd->vkFreeMemory(m_vkd->device(), memory, nullptr);
        return DxvkMemory();
      }
    }

    return DxvkMemory(this, VK_NULL_HANDLE, memory, 0, req->size, mapPtr,
      /* isRaw = */ true, memTypeIndex);
  }


  VmaPool DxvkMemoryAllocator::findOrCreatePool(
          DxvkMemoryPoolCategory           category,
          uint32_t                         memTypeIndex) {
    // Only ever called from tryAlloc(), which always runs under m_mutex,
    // so the map access here doesn't need its own lock.
    auto& poolsOfCategory = m_pools[uint32_t(category)];

    auto it = poolsOfCategory.find(memTypeIndex);
    if (it != poolsOfCategory.end())
      return it->second;

    VmaPoolCreateInfo poolInfo = {};
    poolInfo.memoryTypeIndex = memTypeIndex;
    poolInfo.flags           = 0;

    // Default (tuned-by-testing) block sizes for a "normal-sized" heap.
    switch (category) {
      case DxvkMemoryPoolCategory::Small:
        // Small/transient churn: keep blocks cheap to create and free.
        poolInfo.blockSize = 16ull << 20;
        break;
      case DxvkMemoryPoolCategory::Resident:
        // Long-lived GPU resources: bigger blocks, fewer of them.
        poolInfo.blockSize = 128ull << 20;
        break;
      case DxvkMemoryPoolCategory::Generic:
      default:
        // 0 = let VMA use the allocator-wide preferredLargeHeapBlockSize.
        poolInfo.blockSize = 0;
        break;
    }

    // Scale down for unusually small heaps -- low-VRAM and many UMA/iGPU
    // devices can expose a heap far smaller than the 128 MB "Resident"
    // default above, especially after the UMA-specific 70/80% cap applied
    // in the constructor. Without this, a single pool block could end up
    // requesting a large fraction (or more) of the entire usable heap on
    // its own. Left untouched on any reasonably-sized heap, since the
    // defaults above were chosen based on real testing (see constructor).
    if (poolInfo.blockSize != 0) {
      VkDeviceSize heapSize = m_memProps.memoryHeaps[
        m_memProps.memoryTypes[memTypeIndex].heapIndex].size;

      if (heapSize != 0) {
        constexpr VkDeviceSize maxHeapFraction = 8; // at most 1/8th of the heap
        poolInfo.blockSize = std::min(poolInfo.blockSize, heapSize / maxHeapFraction);
        poolInfo.blockSize = std::max(poolInfo.blockSize, VkDeviceSize(1ull << 20));
      }
    }

    VmaPool pool = VK_NULL_HANDLE;

    if (vmaCreatePool(m_vma, &poolInfo, &pool) != VK_SUCCESS)
      return VK_NULL_HANDLE;

    poolsOfCategory.emplace(memTypeIndex, pool);
    return pool;
  }


  DxvkMemoryStats DxvkMemoryAllocator::getMemoryStats(uint32_t heap) const {
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets = { };
    vmaGetHeapBudgets(m_vma, budgets.data());

    return computeMemoryStats(budgets[heap],
      m_rawAllocated[heap].load(std::memory_order_relaxed));
  }


  void DxvkMemoryAllocator::notifyMemoryAlloc(uint32_t memoryTypeIndex, VkDeviceSize size) {
    uint32_t heapIndex = m_memProps.memoryTypes[memoryTypeIndex].heapIndex;
    m_device->adapter()->notifyHeapMemoryAlloc(heapIndex, size);
  }


  void DxvkMemoryAllocator::notifyMemoryFree(uint32_t memoryTypeIndex, VkDeviceSize size) {
    uint32_t heapIndex = m_memProps.memoryTypes[memoryTypeIndex].heapIndex;
    m_device->adapter()->notifyHeapMemoryFree(heapIndex, size);
  }


  void DxvkMemoryAllocator::notifyRawAlloc(uint32_t memoryTypeIndex, VkDeviceSize size) {
    uint32_t heapIndex = m_memProps.memoryTypes[memoryTypeIndex].heapIndex;
    m_rawAllocated[heapIndex] += size;
    m_device->adapter()->notifyHeapMemoryAlloc(heapIndex, size);
  }


  void DxvkMemoryAllocator::notifyRawFree(uint32_t memoryTypeIndex, VkDeviceSize size) {
    uint32_t heapIndex = m_memProps.memoryTypes[memoryTypeIndex].heapIndex;
    m_rawAllocated[heapIndex] -= size;
    m_device->adapter()->notifyHeapMemoryFree(heapIndex, size);
  }


  VKAPI_ATTR void VKAPI_CALL DxvkMemoryAllocator::deviceMemoryAllocateCallback(
          VmaAllocator      /* allocator */,
          uint32_t          memoryType,
          VkDeviceMemory    /* memory */,
          VkDeviceSize      size,
          void*             pUserData) {
    // Fires once per real vkAllocateMemory call VMA makes internally --
    // i.e. once per block, not once per suballocation -- which is exactly
    // the granularity the old free-list allocator's notifyHeapMemoryAlloc
    // calls used to run at.
    static_cast<DxvkMemoryAllocator*>(pUserData)->notifyMemoryAlloc(memoryType, size);
  }


  VKAPI_ATTR void VKAPI_CALL DxvkMemoryAllocator::deviceMemoryFreeCallback(
          VmaAllocator      /* allocator */,
          uint32_t          memoryType,
          VkDeviceMemory    /* memory */,
          VkDeviceSize      size,
          void*             pUserData) {
    static_cast<DxvkMemoryAllocator*>(pUserData)->notifyMemoryFree(memoryType, size);
  }

}
