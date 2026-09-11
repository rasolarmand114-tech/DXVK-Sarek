#pragma once

#include <array>
#include <atomic>
#include <unordered_map>

#include "dxvk_adapter.h"

// Guard against VMA_IMPLEMENTATION being defined project-wide (e.g. via
// a global compiler flag in the build system) and pulling the full VMA
// function bodies into every translation unit that includes this header.
// Only dxvk_memory.cpp is allowed to provide the real implementation,
// signalled by VMA_DXVK_IMPLEMENTATION; everyone else only ever gets
// declarations, which is what a single-header library like VMA requires
// to avoid duplicate-symbol errors at link time.
#if defined(VMA_IMPLEMENTATION) && !defined(VMA_DXVK_IMPLEMENTATION)
  #undef VMA_IMPLEMENTATION
#endif

// This project loads all Vulkan entry points dynamically through its own
// vk::InstanceFn / vk::DeviceFn wrappers and never links against the
// Vulkan loader (vulkan-1 / libvulkan) directly. By default VMA expects
// to either link those symbols statically (VMA_STATIC_VULKAN_FUNCTIONS)
// or fetch them itself via vkGetInstanceProcAddr/vkGetDeviceProcAddr
// (VMA_DYNAMIC_VULKAN_FUNCTIONS), both of which fail for us at link time.
// Disabling both forces VMA to use exclusively the function pointers we
// hand it through VmaAllocatorCreateInfo::pVulkanFunctions.
#ifndef VMA_STATIC_VULKAN_FUNCTIONS
  #define VMA_STATIC_VULKAN_FUNCTIONS 0
#endif

#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
  #define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#endif

#include "../vulkan/vk_mem_alloc.h" // = vk_mem_alloc.h

namespace dxvk {

  class DxvkMemoryAllocator;

  /**
   * \brief Memory stats
   *
   * Reports the amount of device memory
   * allocated and used by the application.
   * Combines VMA's per-heap budget statistics
   * with byte counts for allocations that
   * bypass VMA entirely (see DxvkMemory::isRaw),
   * which VMA has no visibility into.
   */
  struct DxvkMemoryStats {
    VkDeviceSize memoryAllocated = 0;
    VkDeviceSize memoryUsed      = 0;
  };


  enum class DxvkSharedHandleMode {
      None,
      Import,
      Export,
  };

  /**
   * \brief Shared handle info
   *
   * The shared resource information for a given resource.
   */
  struct DxvkSharedHandleInfo {
    DxvkSharedHandleMode mode = DxvkSharedHandleMode::None;
    VkExternalMemoryHandleTypeFlagBits type   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_FLAG_BITS_MAX_ENUM;
    union {
#ifdef _WIN32
      HANDLE                             handle = INVALID_HANDLE_VALUE;
#else
      // Placeholder for other handle types, such as FD
      void *dummy;
#endif
    };
  };


  /**
   * \brief Memory allocation flags
   *
   * Used as hints when picking a VMA allocation strategy /
   * priority. VMA does its own chunk sub-allocation, so these
   * no longer gate which chunk an allocation can land in the
   * way the old free-list allocator's hints did; they now only
   * influence VMA allocation priority (VK_EXT_memory_priority)
   * and pooling behaviour.
   */
  enum class DxvkMemoryFlag : uint32_t {
    Small             = 0,  ///< Small allocation
    GpuReadable       = 1,  ///< Medium-priority resource
    GpuWritable       = 2,  ///< High-priority resource
    Transient         = 3,  ///< Resource is short-lived
    IgnoreConstraints = 4,  ///< Ignore most allocation flags
  };

  using DxvkMemoryFlags = Flags<DxvkMemoryFlag>;


  /**
   * \brief Coarse allocation-lifetime categories
   *
   * Each category gets its own set of VMA pools (one pool per Vulkan
   * memory type actually used within that category), each with a block
   * size tuned for how that category behaves, instead of every resource
   * in the app sharing one pile of generic blocks:
   *  - Small: high churn, short-lived, small allocations -> small blocks,
   *    so creating a fresh block is cheap and doesn't waste much space.
   *  - Resident: long-lived GPU-priority resources (textures, RTs, big
   *    buffers) -> large blocks, packed tightly since they stick around.
   *  - Generic: everything else (host-visible staging, unclassified).
   */
  enum class DxvkMemoryPoolCategory : uint32_t {
    Small    = 0,
    Resident = 1,
    Generic  = 2,
  };

  constexpr uint32_t DxvkMemoryPoolCategoryCount = 3;


  /**
   * \brief Memory slice
   *
   * Wraps a single VMA allocation (\c VmaAllocation). Memory is
   * returned to VMA automatically via \c vmaFreeMemory once the
   * last reference to the slice goes out of scope.
   */
  class DxvkMemory {
    friend class DxvkMemoryAllocator;
  public:

    DxvkMemory();
    DxvkMemory(
            DxvkMemoryAllocator*  alloc,
            VmaAllocation         allocation,
            VkDeviceMemory        memory,
            VkDeviceSize          offset,
            VkDeviceSize          length,
            void*                 mapPtr,
            bool                  isRaw = false,
            uint32_t              memoryTypeIndex = 0);
    DxvkMemory             (DxvkMemory&& other);
    DxvkMemory& operator = (DxvkMemory&& other);
    ~DxvkMemory();

    /**
     * \brief Memory object
     *
     * This information is required when
     * binding memory to Vulkan objects.
     * \returns Memory object
     */
    VkDeviceMemory memory() const {
      return m_memory;
    }

    /**
     * \brief Offset into device memory
     *
     * This information is required when
     * binding memory to Vulkan objects.
     * \returns Offset into device memory
     */
    VkDeviceSize offset() const {
      return m_offset;
    }

    /**
     * \brief Pointer to mapped data
     *
     * \param [in] offset Byte offset
     * \returns Pointer to mapped data
     */
    void* mapPtr(VkDeviceSize offset) const {
      return reinterpret_cast<char*>(m_mapPtr) + offset;
    }

    /**
     * \brief Returns length of memory allocated
     *
     * \returns Memory size
     */
    VkDeviceSize length() const {
      return m_length;
    }

    /**
     * \brief Checks whether the memory slice is defined
     *
     * \returns \c true if this slice points to actual device
     *          memory (whether VMA-managed or raw), \c false
     *          if it is undefined.
     */
    operator bool () const {
      return m_memory != VK_NULL_HANDLE;
    }

    /**
     * \brief Underlying VMA allocation handle
     *
     * Exposed in case callers need to pass it directly
     * to other VMA functions (e.g. flush/invalidate).
     * \c VK_NULL_HANDLE for raw (non-VMA) allocations,
     * see \ref isRaw.
     */
    VmaAllocation allocation() const {
      return m_allocation;
    }

    /**
     * \brief Whether this slice bypasses VMA
     *
     * True for allocations that had to go through raw
     * \c vkAllocateMemory instead of VMA â€” currently only
     * shared/external memory (import/export), since VMA's
     * public API has no way to carry the extra \c pNext
     * chain (\c VkExportMemoryAllocateInfo /
     * \c VkImportMemoryWin32HandleInfoKHR) those need.
     */
    bool isRaw() const {
      return m_isRaw;
    }

  private:

    DxvkMemoryAllocator*  m_alloc           = nullptr;
    VmaAllocation         m_allocation      = VK_NULL_HANDLE;
    VkDeviceMemory        m_memory          = VK_NULL_HANDLE;
    VkDeviceSize          m_offset          = 0;
    VkDeviceSize          m_length          = 0;
    void*                 m_mapPtr          = nullptr;
    bool                  m_isRaw           = false;
    // Only meaningful when m_isRaw is true: needed at free() time so we
    // can correctly attribute the freed bytes back to the heap they came
    // from (see DxvkMemoryAllocator::notifyRawAlloc/notifyRawFree).
    uint32_t              m_memoryTypeIndex = 0;

    void free();

  };


  /**
   * \brief Memory allocator
   *
   * Thin wrapper around the Vulkan Memory Allocator (VMA).
   * VMA takes care of chunk sub-allocation, placement and
   * memory budget tracking internally, so DXVK no longer
   * needs its own free-list based chunk allocator.
   */
  class DxvkMemoryAllocator {
    friend class DxvkMemory;

    constexpr static VkDeviceSize SmallAllocationThreshold = 256 << 10;
  public:

    DxvkMemoryAllocator(const DxvkDevice* device);
    ~DxvkMemoryAllocator();

    /**
     * \brief Buffer-image granularity
     *
     * The granularity between linear and non-linear
     * resources in adjacent memory locations. See
     * section 11.6 of the Vulkan spec for details.
     * \returns Buffer-image granularity
     */
    VkDeviceSize bufferImageGranularity() const {
      return m_devProps.limits.bufferImageGranularity;
    }

    /**
     * \brief Allocates device memory
     *
     * \param [in] req Memory requirements
     * \param [in] dedAllocReq Dedicated allocation requirements
     * \param [in] dedAllocInfo Dedicated allocation info
     * \param [in] flags Memory type flags
     * \param [in] hints Memory hints
     * \returns Allocated memory slice
     */
    DxvkMemory alloc(
      const VkMemoryRequirements*             req,
      const VkMemoryDedicatedRequirements&    dedAllocReq,
      const VkMemoryDedicatedAllocateInfo&    dedAllocInfo,
            VkMemoryPropertyFlags             flags,
            DxvkMemoryFlags                   hints);

    /**
     * \brief Queries memory stats
     *
     * Returns the total amount of memory
     * allocated and used for a given heap,
     * as reported by VMA's budget query plus
     * any raw (non-VMA) allocations on that heap.
     * \param [in] heap Heap index
     * \returns Memory stats for this heap
     */
    DxvkMemoryStats getMemoryStats(uint32_t heap) const;

    /**
     * \brief Underlying VMA allocator handle
     */
    VmaAllocator handle() const {
      return m_vma;
    }

  private:

    const Rc<vk::DeviceFn>                 m_vkd;
    const DxvkDevice*                      m_device;
    const VkPhysicalDeviceProperties       m_devProps;
    const VkPhysicalDeviceMemoryProperties m_memProps;

    // Cached once at construction time (DxvkDevice::isUnifiedMemoryArchitecture()
    // re-queries memory properties on every call) since several allocation
    // decisions below key off this.
    const bool m_isUMA;

    VmaAllocator m_vma = VK_NULL_HANDLE;

    // Restored after testing: removing this made things noticeably worse
    // (severe hitching, near-crashes) rather than better. VMA's internal
    // per-pool locking is fine for correctness, but without any throttling
    // here, bursts of concurrent vkAllocateMemory calls from multiple
    // threads at once (render thread + streaming/loader threads all
    // needing fresh blocks at the same moment) appear to overwhelm the
    // driver/kernel far worse than a single serialized queue does. Keeping
    // the mutex trades a bit of contention for actual stability.
    dxvk::mutex m_mutex;

    // One pool cache per category (see DxvkMemoryPoolCategory), keyed by
    // Vulkan memory type index within that category. Only ever touched
    // from inside tryAlloc(), which always runs under m_mutex, so this
    // doesn't need its own lock.
    std::array<std::unordered_map<uint32_t, VmaPool>, DxvkMemoryPoolCategoryCount> m_pools;

    // Per-heap byte counters for allocations that bypass VMA (see
    // DxvkMemory::isRaw). VMA's own vmaGetHeapBudgets() has no idea these
    // exist, so getMemoryStats() adds this on top of what VMA reports.
    // Updated from tryAllocRaw()/DxvkMemory::free(), which may run
    // concurrently on different threads without holding m_mutex (free()
    // in particular is never synchronized by it), hence atomic. Not
    // using a default member initializer here: std::atomic's default
    // constructor isn't guaranteed to zero-initialize on every standard
    // this project might be compiled against, so this is explicitly
    // zeroed in the constructor body instead (same reasoning as
    // DxvkAdapter::initHeapAllocInfo() for its own atomic array).
    std::array<std::atomic<VkDeviceSize>, VK_MAX_MEMORY_HEAPS> m_rawAllocated;

    VmaPool findOrCreatePool(
            DxvkMemoryPoolCategory           category,
            uint32_t                         memTypeIndex);

    DxvkMemory tryAlloc(
      const VkMemoryRequirements*             req,
      const VkMemoryDedicatedRequirements&    dedAllocReq,
      const VkMemoryDedicatedAllocateInfo&    dedAllocInfo,
            VkMemoryPropertyFlags             flags,
            DxvkMemoryFlags                   hints);

    // Bypasses VMA entirely: used only when dedAllocInfo carries an extra
    // pNext chain (shared/external memory) that VMA's public API can't
    // be handed. See the comment at the top of tryAlloc() in the .cpp.
    DxvkMemory tryAllocRaw(
      const VkMemoryRequirements*             req,
      const VkMemoryDedicatedAllocateInfo&    dedAllocInfo,
            VkMemoryPropertyFlags             flags,
            DxvkMemoryFlags                   hints);

    // Heap-accounting helpers. notifyMemoryAlloc/Free mirror every real
    // vkAllocateMemory/vkFreeMemory call VMA makes on our behalf back to
    // DxvkAdapter (see VmaDeviceMemoryCallbacks in the constructor), so
    // DxvkAdapter::getMemoryHeapInfo() stays accurate even on devices
    // without VK_EXT_memory_budget. notifyRawAlloc/Free do the same for
    // allocations that bypass VMA (tryAllocRaw), and additionally update
    // m_rawAllocated so getMemoryStats() can account for them too.
    void notifyMemoryAlloc(uint32_t memoryTypeIndex, VkDeviceSize size);
    void notifyMemoryFree (uint32_t memoryTypeIndex, VkDeviceSize size);

    void notifyRawAlloc(uint32_t memoryTypeIndex, VkDeviceSize size);
    void notifyRawFree (uint32_t memoryTypeIndex, VkDeviceSize size);

    static VKAPI_ATTR void VKAPI_CALL deviceMemoryAllocateCallback(
      VmaAllocator      allocator,
      uint32_t          memoryType,
      VkDeviceMemory    memory,
      VkDeviceSize      size,
      void*             pUserData);

    static VKAPI_ATTR void VKAPI_CALL deviceMemoryFreeCallback(
      VmaAllocator      allocator,
      uint32_t          memoryType,
      VkDeviceMemory    memory,
      VkDeviceSize      size,
      void*             pUserData);

  };

}
